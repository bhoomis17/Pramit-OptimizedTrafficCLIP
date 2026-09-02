import numpy as np


def calculate_mean_iat(packets):
    """
    Calculate the mean inter-arrival time (IAT)
    of packets in a flow.

    Returns:
        float: Mean IAT in milliseconds.
    """

    if len(packets) < 2:
        return 0.0

    timestamps = np.array(
        [float(packet.time) for packet in packets],
        dtype=np.float64
    )

    iats = np.diff(timestamps) * 1000.0

    return float(np.mean(iats))


def calculate_jitter(packets):
    """
    Calculate jitter as the standard deviation
    of inter-arrival times.

    Returns:
        float: Jitter in milliseconds.
    """

    if len(packets) < 2:
        return 0.0

    timestamps = np.array(
        [float(packet.time) for packet in packets],
        dtype=np.float64
    )

    iats = np.diff(timestamps) * 1000.0

    return float(np.std(iats))


def calculate_entropy(packets):
    """
    Calculate Shannon entropy of the payload bytes
    in a flow.

    Returns:
        float: Shannon entropy.
    """

    payload_bytes = []

    for packet in packets:

        # Raw packet payload
        if hasattr(packet, "payload"):
            try:
                payload_bytes.extend(bytes(packet.payload))
            except Exception:
                pass

    if len(payload_bytes) == 0:
        return 0.0

    data = np.array(payload_bytes, dtype=np.uint8)

    counts = np.bincount(data, minlength=256)

    probabilities = counts / np.sum(counts)

    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


def calculate_statistics(packets):
    """
    Calculate all three statistical traffic features.

    Returns:
        numpy.ndarray:
        [mean_iat, jitter, entropy]
    """

    mean_iat = calculate_mean_iat(packets)
    jitter = calculate_jitter(packets)
    entropy = calculate_entropy(packets)

    return np.array(
        [mean_iat, jitter, entropy],
        dtype=np.float32
    )
