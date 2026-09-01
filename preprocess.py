import argparse
import os
import csv

from src.preprocessing.pcap_reader import read_pcap
from src.preprocessing.flow_extractor import extract_flows
from src.preprocessing.pcap_to_image import packets_to_image
from src.preprocessing.statistics import calculate_statistics


def main():
    parser = argparse.ArgumentParser(
        description="Convert PCAP flows into traffic images"
    )

    parser.add_argument(
        "--pcap",
        required=True,
        help="Path to the PCAP file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory where generated images will be saved"
    )

    args = parser.parse_args()

    # Create output directory if it does not exist
    os.makedirs(args.output, exist_ok=True)

    print("===================================")
    print("TrafficCLIP Preprocessing")
    print("===================================")
    print(f"PCAP: {args.pcap}")
    print(f"Output: {args.output}")

    # Step 1: Read PCAP
    print("\n[1/3] Reading PCAP...")

    packets = read_pcap(args.pcap)

    print(f"Packets loaded: {len(packets)}")

    # Step 2: Extract flows
    print("\n[2/3] Extracting flows...")

    flows = extract_flows(packets)

    print(f"Flows found: {len(flows)}")

    # Step 3: Convert each flow into an image
    # and calculate statistics
    print("\n[3/3] Creating traffic images and calculating statistics...")

    statistics_path = os.path.join(
        args.output,
        "statistics.csv"
    )

    with open(
        statistics_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "image_path",
            "mean_iat",
            "jitter",
            "entropy"
        ])

        for index, flow_packets in enumerate(flows.values()):

            # Create traffic image
            image = packets_to_image(flow_packets)

            output_path = os.path.join(
                args.output,
                f"flow_{index:06d}.png"
            )

            image.save(output_path)

            # Calculate statistical features
            stats = calculate_statistics(flow_packets)

            mean_iat = stats[0]
            jitter = stats[1]
            entropy = stats[2]

            # Save statistics associated with this image
            writer.writerow([
                output_path,
                mean_iat,
                jitter,
                entropy
            ])

    print("\n===================================")
    print("Preprocessing completed!")
    print(f"Images saved to: {args.output}")
    print(f"Statistics saved to: {statistics_path}")
    print("===================================")


if __name__ == "__main__":
    main()