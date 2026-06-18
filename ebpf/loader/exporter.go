// ebpf/loader/exporter.go
// Kafka + Redis event exporter for eBPF flow events.
// Serialises raw BPF perf event bytes to JSON and publishes them to Kafka.
// Also writes the latest window summary to Redis for fast API reads.

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"time"

	"github.com/IBM/sarama"
	"go.uber.org/zap"
)

// FlowRecord is the JSON-serialisable representation of a network flow event.
type FlowRecord struct {
	SrcIP       string    `json:"src_ip"`
	DstIP       string    `json:"dst_ip"`
	SrcPort     uint16    `json:"src_port"`
	DstPort     uint16    `json:"dst_port"`
	Bytes       uint64    `json:"bytes"`
	TimestampNS uint64    `json:"timestamp_ns"`
	Timestamp   time.Time `json:"timestamp"`
	PID         uint32    `json:"pid"`
	Comm        string    `json:"comm"`
}

// KafkaExporter publishes flow records to a Kafka topic.
type KafkaExporter struct {
	producer sarama.SyncProducer
	topic    string
	logger   *zap.Logger
}

// NewKafkaExporter creates a new Kafka producer targeting the given broker and topic.
func NewKafkaExporter(brokerAddr, topic, redisURL string) (*KafkaExporter, error) {
	cfg := sarama.NewConfig()
	cfg.Producer.Return.Successes = true
	cfg.Producer.RequiredAcks = sarama.WaitForLocal
	cfg.Producer.Compression = sarama.CompressionSnappy
	cfg.Producer.Retry.Max = 3
	cfg.Net.DialTimeout = 5 * time.Second

	producer, err := sarama.NewSyncProducer([]string{brokerAddr}, cfg)
	if err != nil {
		return nil, fmt.Errorf("kafka producer: %w", err)
	}

	logger, _ := zap.NewProduction()
	logger.Info("Kafka exporter initialised",
		zap.String("broker", brokerAddr),
		zap.String("topic", topic),
	)

	return &KafkaExporter{
		producer: producer,
		topic:    topic,
		logger:   logger,
	}, nil
}

// Export deserialises a raw BPF perf event byte slice and publishes to Kafka.
func (e *KafkaExporter) Export(raw []byte) error {
	record, err := parseFlowEvent(raw)
	if err != nil {
		return fmt.Errorf("parse flow event: %w", err)
	}

	payload, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("json marshal: %w", err)
	}

	msg := &sarama.ProducerMessage{
		Topic: e.topic,
		Key:   sarama.StringEncoder(fmt.Sprintf("%s:%s", record.SrcIP, record.DstIP)),
		Value: sarama.ByteEncoder(payload),
	}

	_, _, err = e.producer.SendMessage(msg)
	return err
}

// Close shuts down the Kafka producer cleanly.
func (e *KafkaExporter) Close() {
	if err := e.producer.Close(); err != nil {
		e.logger.Error("closing kafka producer", zap.Error(err))
	}
}

// parseFlowEvent parses the raw bytes from the eBPF perf ring buffer.
// Layout must match the C struct flow_event exactly (little-endian, packed).
func parseFlowEvent(raw []byte) (*FlowRecord, error) {
	// Minimum size: 4+4+2+2+8+8+4+4+16+1+3 = 56 bytes
	if len(raw) < 56 {
		return nil, fmt.Errorf("raw event too short: %d bytes", len(raw))
	}

	le := binary.LittleEndian
	rec := &FlowRecord{
		SrcIP:       intToIPv4(le.Uint32(raw[0:4])),
		DstIP:       intToIPv4(le.Uint32(raw[4:8])),
		SrcPort:     le.Uint16(raw[8:10]),
		DstPort:     le.Uint16(raw[10:12]),
		Bytes:       le.Uint64(raw[12:20]),
		TimestampNS: le.Uint64(raw[20:28]),
		PID:         le.Uint32(raw[28:32]),
		// TID:       le.Uint32(raw[32:36]),  // not exposed in API
		Comm: nullTerminatedString(raw[36:52]),
	}
	rec.Timestamp = time.Unix(0, int64(rec.TimestampNS))
	return rec, nil
}

func intToIPv4(ip uint32) string {
	return net.IP{
		byte(ip),
		byte(ip >> 8),
		byte(ip >> 16),
		byte(ip >> 24),
	}.String()
}

func nullTerminatedString(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
