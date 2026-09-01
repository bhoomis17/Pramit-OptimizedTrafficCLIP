import numpy as np


def calculate_mean_iat(packet_timestamps):
    """
    Calculate Mean Inter-Arrival Time (IAT).

    Parameters
    ----------
    packet_timestamps : array-like
        Packet timestamps in seconds.

    Returns
    -------
    float
        Mean IAT in milliseconds.
    """

    timestamps = np.asarray(packet_timestamps, dtype=float)

    if len(timestamps) < 2:
        return 0.0

    # Difference between consecutive packet timestamps
    iats = np.diff(timestamps)

    # Convert seconds to milliseconds
    iats_ms = iats * 1000.0

    return float(np.mean(iats_ms))


def calculate_jitter(packet_timestamps):
    """
    Calculate jitter as the standard deviation of IATs.

    Parameters
    ----------
    packet_timestamps : array-like
        Packet timestamps in seconds.

    Returns
    -------
    float
        Jitter in milliseconds.
    """

    timestamps = np.asarray(packet_timestamps, dtype=float)

    if len(timestamps) < 2:
        return 0.0

    iats = np.diff(timestamps)

    # Convert seconds to milliseconds
    iats_ms = iats * 1000.0

    return float(np.std(iats_ms))


def calculate_entropy(payload):
    """
    Calculate Shannon entropy of packet payload bytes.

    Parameters
    ----------
    payload : array-like
        Payload bytes with values in the range 0-255.

    Returns
    -------
    float
        Shannon entropy in bits.
    """

    payload = np.asarray(payload, dtype=np.uint8).flatten()

    if len(payload) == 0:
        return 0.0

    # Count occurrences of each byte value
    counts = np.bincount(payload, minlength=256)

    # Convert counts to probabilities
    probabilities = counts / np.sum(counts)

    # Remove zero probabilities to avoid log2(0)
    probabilities = probabilities[probabilities > 0]

    # Shannon entropy
    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


def calculate_statistics(packet_timestamps, payload):
    """
    Calculate all three traffic statistics.

    Returns
    -------
    numpy.ndarray
        [mean_iat, jitter, entropy]
    """

    mean_iat = calculate_mean_iat(packet_timestamps)
    jitter = calculate_jitter(packet_timestamps)
    entropy = calculate_entropy(payload)

    return np.array(
        [mean_iat, jitter, entropy],
        dtype=np.float32
    )


if __name__ == "__main__":
    # Simple test data

    timestamps = np.array([
        0.000,
        0.010,
        0.025,
        0.045,
        0.070
    ])

    payload = np.array([
        10, 20, 10, 20, 30,
        10, 20, 30, 40, 50
    ])

    mean_iat = calculate_mean_iat(timestamps)
    jitter = calculate_jitter(timestamps)
    entropy = calculate_entropy(payload)

    statistics = calculate_statistics(
        timestamps,
        payload
    )

    print("Mean IAT:", mean_iat, "ms")
    print("Jitter:", jitter, "ms")
    print("Entropy:", entropy)
    print("Statistics vector:", statistics)