export type TwinStatus =
  | "ONLINE"
  | "OFFLINE"
  | "SUSPICIOUS"
  | "HIGH_RISK"
  | "POTENTIAL_ATTACK";

export type RiskLevel =
  | "LOW"
  | "MEDIUM"
  | "HIGH"
  | "CRITICAL";

export type DeviceSource =
  | "REAL"
  | "VIRTUAL";

export type ConnectionProtocol =
  | "MQTT"
  | "TCP"
  | "UDP"
  | "HTTP"
  | "HTTPS";

export interface DeviceTwinState {
  deviceId: string;
  deviceType: "ESP32";

  /**
   * REAL = physical ESP32
   * VIRTUAL = simulated IoT device
   */
  source: DeviceSource;

  status: TwinStatus;

  ipAddress: string;

  firmwareVersion: string;

  uptimeSeconds: number;

  // Physical/device state
  temperature: number;
  humidity: number;
  voltage: number;
  wifiRssi: number;

  // Network state
  mqttConnected: boolean;
  messagesPerSecond: number;
  bytesPerSecond: number;
  activeConnections: number;

  // Security state
  anomalyScore: number;
  riskLevel: RiskLevel;
  attackDetected: boolean;

  lastSeen: string;
}

/**
 * Type-only compatibility for dormant gateway presentation components.
 * No Raspberry Pi integration is enabled or implemented by this project.
 */
export interface GatewayTwinState {
  deviceId: string;
  status: TwinStatus;
  ipAddress: string;
  mqttConnected: boolean;
  connectedDevices: number;
  messagesPerSecond: number;
  bytesPerSecond: number;
  anomalyScore: number;
  riskLevel: RiskLevel;
}

export interface NetworkConnection {
  id: string;

  source: string;
  destination: string;

  protocol: ConnectionProtocol;

  status: "ACTIVE" | "INACTIVE" | "SUSPICIOUS";

  messagesPerSecond: number;
  bytesPerSecond: number;

  lastActivity: string;
}

export interface AttackerNode {
  id: string;

  name: string;

  status: "ACTIVE" | "INACTIVE";

  sourceIp: string;

  targetDevice: string;

  attackType: AttackType;

  intensity: number;

  startedAt: string | null;
}

export type AttackType =
  | "NORMAL"
  | "GAFGYT_COMBO"
  | "MIRAI_ACK"
  | "MIRAI_SCAN"
  | "MIRAI_SYN"
  | "MIRAI_UDP"
  | "MIRAI_UDPPLAIN";

export interface AttackEvent {
  id: string;

  attackerId: string;

  targetDevice: string;

  attackType: AttackType;

  severity: RiskLevel;

  intensity: number;

  timestamp: string;

  detected: boolean;

  anomalyScore: number;

  description: string;

  // Real ML result
  prediction?: "BENIGN" | "ATTACK";

  confidenceScore?: number;

  modelName?: string;
}

export interface DigitalTwinState {
  devices: DeviceTwinState[];

  connections: NetworkConnection[];

  attackers: AttackerNode[];

  attackEvents: AttackEvent[];

  networkHealth: number;

  globalRiskScore: number;

  activeAttacks: number;
}
