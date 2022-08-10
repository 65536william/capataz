from utils import get_sequences
import os
import torch


def save_sequences(file_paths: str, args):
    all_sequences = get_sequences(file_paths, args)
    for idx, sequence_group in enumerate(all_sequences):
        total_sequence_group_len = len(sequence_group)
        if args["output_format"] == "pt":
            new_file_path = os.path.join(
                args["output_directory"],
                f"{args['output_name']}_{idx}_{total_sequence_group_len}.pt",
            )
            print("writing to drive")
            torch.save(torch.tensor(sequence_group, dtype=torch.float16), new_file_path)
            print(f"{new_file_path} saved")
        elif args["output_format"] == "tfrecords":
            new_file_path = os.path.join(
                args.output_dir, f"{args.name}_{idx}_{total_chunk_len}.tfrecords"
            )
            print("writing to drive")
            with tf.io.TFRecordWriter(new_file_path) as writer:
                for seq in chunk_group:
                    feature = {
                        "text": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=seq)
                        )
                    }
                    tf_example = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    writer.write(tf_example.SerializeToString())
            print(f"{new_file_path} saved")
