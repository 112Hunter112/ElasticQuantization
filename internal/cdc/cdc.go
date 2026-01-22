package cdc

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/jackc/pglogrepl"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgproto3/v2"
	"github.com/yourusername/consistency-auditor/internal/config"
)

type EventType string

const (
	Insert EventType = "INSERT"
	Update EventType = "UPDATE"
	Delete EventType = "DELETE"
)

type Event struct {
	Type  EventType
	Table string
	ID    string
	Field string // Optional: changed field for partial updates
	Data  map[string]interface{}
}

type CDCListener struct {
	config    config.CDCConfig
	dbConfig  config.DatabaseConfig
	eventChan chan Event
	stopChan  chan struct{}
	conn      *pgconn.PgConn
	relations map[uint32]*pglogrepl.RelationMessage
	walPos    pglogrepl.LSN
}

func NewCDCListener(cfg config.CDCConfig, dbCfg config.DatabaseConfig) *CDCListener {
	return &CDCListener{
		config:    cfg,
		dbConfig:  dbCfg,
		eventChan: make(chan Event, 1000), // Buffered channel
		stopChan:  make(chan struct{}),
		relations: make(map[uint32]*pglogrepl.RelationMessage),
	}
}

func (l *CDCListener) Start() error {
	connConfig, err := pgconn.ParseConfig(fmt.Sprintf(
		"postgres://%s:%s@%s:%d/%s?replication=database",
		l.dbConfig.User, l.dbConfig.Password, l.dbConfig.Host, l.dbConfig.Port, l.dbConfig.Database,
	))
	if err != nil {
		return fmt.Errorf("failed to parse config: %w", err)
	}

	conn, err := pgconn.ConnectConfig(context.Background(), connConfig)
	if err != nil {
		return fmt.Errorf("failed to connect to postgres: %w", err)
	}
	l.conn = conn

	// Create replication slot if it doesn't exist
	// We ignore the error "already exists"
	_, err = pglogrepl.CreateReplicationSlot(context.Background(), l.conn, l.config.SlotName, "pgoutput", pglogrepl.CreateReplicationSlotOptions{Temporary: false})
	if err != nil {
		if !strings.Contains(err.Error(), "already exists") && !strings.Contains(err.Error(), "SQLSTATE 42710") {
			log.Printf("Warning: Failed to create replication slot: %v", err)
		}
	}

	log.Printf("Starting logical replication on slot %s", l.config.SlotName)
	err = pglogrepl.StartReplication(context.Background(), l.conn, l.config.SlotName, 0, pglogrepl.StartReplicationOptions{
		PluginArgs: []string{"proto_version '1'", fmt.Sprintf("publication_names '%s'", l.config.Publication)},
	})
	if err != nil {
		return fmt.Errorf("failed to start replication: %w", err)
	}

	go l.listen()
	return nil
}

func (l *CDCListener) listen() {
	defer func() {
		if l.conn != nil {
			l.conn.Close(context.Background())
		}
	}()

	standbyMessageTimeout := time.Second * 10
	nextStandbyMessageDeadline := time.Now().Add(standbyMessageTimeout)

	for {
		select {
		case <-l.stopChan:
			return
		default:
			if time.Now().After(nextStandbyMessageDeadline) {
				err := pglogrepl.SendStandbyStatusUpdate(context.Background(), l.conn, pglogrepl.StandbyStatusUpdate{WALWritePosition: l.walPos})
				if err != nil {
					log.Printf("Failed to send standby status update: %v", err)
				}
				nextStandbyMessageDeadline = time.Now().Add(standbyMessageTimeout)
			}

			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			msg, err := l.conn.ReceiveMessage(ctx)
			cancel()

			if err != nil {
				if pgconn.Timeout(err) {
					continue
				}
				// Check if it's just a closed connection due to Stop()
				select {
				case <-l.stopChan:
					return
				default:
					log.Printf("ReceiveMessage failed: %v", err)
					// In a real production app, add backoff and reconnect logic here
					time.Sleep(5 * time.Second)
					return
				}
			}

			switch msg := msg.(type) {
			case *pgproto3.CopyData:
				switch msg.Data[0] {
				case pglogrepl.PrimaryKeepaliveMessageByteID:
					pkm, err := pglogrepl.ParsePrimaryKeepaliveMessage(msg.Data[1:])
					if err != nil {
						log.Printf("ParsePrimaryKeepaliveMessage failed: %v", err)
						continue
					}
					if pkm.ReplyRequested {
						nextStandbyMessageDeadline = time.Time{}
					}

				case pglogrepl.XLogDataByteID:
					xld, err := pglogrepl.ParseXLogData(msg.Data[1:])
					if err != nil {
						log.Printf("ParseXLogData failed: %v", err)
						continue
					}

					l.processLogicalMsg(xld.WALData)
					l.walPos = xld.WALStart + pglogrepl.LSN(len(xld.WALData))
				}
			default:
				if msg != nil {
					log.Printf("Received unexpected message: %T", msg)
				}
			}
		}
	}
}

func (l *CDCListener) processLogicalMsg(data []byte) {
	logicalMsg, err := pglogrepl.Parse(data)
	if err != nil {
		log.Printf("Parse logical message failed: %v", err)
		return
	}

	switch logicalMsg := logicalMsg.(type) {
	case *pglogrepl.RelationMessage:
		l.relations[logicalMsg.RelationID] = logicalMsg

	case *pglogrepl.InsertMessage:
		rel, ok := l.relations[logicalMsg.RelationID]
		if !ok {
			log.Printf("Unknown relation ID: %d", logicalMsg.RelationID)
			return
		}
		data := l.extractData(rel, logicalMsg.Tuple)
		id := l.extractID(data)
		if id != "" {
			l.eventChan <- Event{Type: Insert, Table: rel.Namespace + "." + rel.RelationName, ID: id, Data: data}
		}

	case *pglogrepl.UpdateMessage:
		rel, ok := l.relations[logicalMsg.RelationID]
		if !ok {
			log.Printf("Unknown relation ID: %d", logicalMsg.RelationID)
			return
		}
		data := l.extractData(rel, logicalMsg.NewTuple)
		id := l.extractID(data)
		if id != "" {
			l.eventChan <- Event{Type: Update, Table: rel.Namespace + "." + rel.RelationName, ID: id, Data: data}
		}

	case *pglogrepl.DeleteMessage:
		rel, ok := l.relations[logicalMsg.RelationID]
		if !ok {
			log.Printf("Unknown relation ID: %d", logicalMsg.RelationID)
			return
		}
		
		// For DELETE, we rely on OldTuple (requires REPLICA IDENTITY FULL)
		var data map[string]interface{}
		if logicalMsg.OldTuple != nil {
			data = l.extractData(rel, logicalMsg.OldTuple)
		}
		id := l.extractID(data)
		if id != "" {
			l.eventChan <- Event{Type: Delete, Table: rel.Namespace + "." + rel.RelationName, ID: id, Data: data}
		}
	}
}

func (l *CDCListener) extractData(rel *pglogrepl.RelationMessage, tuple *pglogrepl.TupleData) map[string]interface{} {
	data := make(map[string]interface{})
	for idx, col := range tuple.Columns {
		if idx >= len(rel.Columns) {
			break
		}
		colName := rel.Columns[idx].Name
		switch col.DataType {
		case 'n': // null
			data[colName] = nil
		case 't': // text
			val := string(col.Data)
			data[colName] = val
		case 'u': // unchanged toast
			// data[colName] = ... (value unchanged)
		}
	}
	return data
}

func (l *CDCListener) extractID(data map[string]interface{}) string {
	if data == nil {
		return ""
	}
	if val, ok := data["id"]; ok && val != nil {
		return fmt.Sprintf("%v", val)
	}
	return ""
}

func (l *CDCListener) Stop() {
	close(l.stopChan)
	if l.conn != nil {
		// Close context to force ReceiveMessage to return if blocked
		// In this simple implementation, we rely on connection closing in defer
	}
	close(l.eventChan)
}

func (l *CDCListener) Events() <-chan Event {
	return l.eventChan
}