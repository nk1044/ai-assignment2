"""Constants for the ABR streaming simulator (see README §4–§5)."""

# Bitrate ladder (kbps), one entry per action. Index = action id.
BITRATES_KBPS = [300, 750, 1200, 1850, 2850, 4300]
A_DIM = len(BITRATES_KBPS)
DEFAULT_ACTION = 1                    # quality of the "previous" chunk at episode start

# Video
CHUNK_LEN_S = 4.0                     # seconds of video per chunk (L)
TOTAL_CHUNKS = 48                     # chunks per video (~3.2 min)

# Player / network
BUFFER_MAX_S = 60.0                   # player stops downloading above this buffer level
SLEEP_STEP_S = 0.5                    # sleep granularity when the buffer is full
LINK_RTT_S = 0.08                     # round-trip time added to every chunk request
PACKET_PAYLOAD_PORTION = 0.95         # fraction of link capacity that carries video bytes
NOISE_LOW, NOISE_HIGH = 0.9, 1.1      # multiplicative noise on download delay

# State (6 x 8 matrix, README §4.1)
S_INFO = 6                            # feature rows
S_LEN = 8                             # history length
BUFFER_NORM_S = 10.0
DELAY_NORM_S = 10.0

# Reward (README §4.4): q(R) mapping and rebuffer penalty mu per QoE variant
REBUF_PENALTY = {"lin": 4.3, "log": 2.66, "hd": 8.0}
HD_REWARD = [1, 2, 3, 12, 15, 20]
