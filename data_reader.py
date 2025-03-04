import json
import logging
import os
import pickle

from prompt import CBAct, CIMAAct, ESConvAct

logger = logging.getLogger(__name__)

role_map = {
    "esc": {"sys": "Therapist", "usr": "Patient"},
    "cima": {"sys": "Teacher", "usr": "Student"},
    "cb": {"sys": "Buyer", "usr": "Seller"},
}
act_map = {"esc": ESConvAct, "cima": CIMAAct, "cb": CBAct}


# Serialize and write a Python object to a file using pickle
def write_pkl(obj, filename):
    # Open file in write-binary mode
    with open(filename, "wb") as f:
        # Serialize object and write to file
        pickle.dump(obj, f)


# Read and deserialize a Python object from a pickle file
def read_pkl(filename):
    # Open file in read-binary mode
    with open(filename, "rb") as f:
        # Deserialize and return object from file
        return pickle.load(f)


# Load and cache dataset features for training or evaluation
def load_and_cache_examples(args, tokenizer, evaluate=False):
    # Determine mode (train or evaluation set)
    mode = args.set_name if evaluate else "train"
    print(mode)

    # Construct path for cached features file
    cached_features_file = os.path.join(
        args.data_dir,
        "sft_{}_{}_{}_{}".format(
            args.data_name,
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Check if cached features exist
    if os.path.exists(cached_features_file):
        # Load features from cache
        logger.info("Loading features from cached file %s", cached_features_file)
        features = read_pkl(cached_features_file)
        print("Loaded number of instance:", len(features["source_ids"]))
    else:
        # Create features from raw dataset if cache doesn't exist
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_to_features(args, tokenizer, mode)
        print("Loaded number of instance:", len(features["source_ids"]))

        # Save features to cache for future use
        logger.info("Saving features into cached file %s", cached_features_file)
        write_pkl(features, cached_features_file)

    # Return loaded or created features
    return features


# Convert raw dataset files into model-ready features
def convert_to_features(args, tokenizer, mode):
    # Construct path to dataset file
    path = os.path.join(args.data_dir, "{}-{}.txt".format(args.data_name, mode))
    # Get sorted list of actions for the dataset type
    act = sorted(list(act_map[args.data_name].keys()))
    print("tokenizing {}".format(path))

    # Open and process dataset file
    with open(path, "r", encoding="utf-8") as infile:
        # Initialize tracking variables
        max_dia_len = 0
        avg_dia_len = []
        source_ids = []
        target_ids = []

        # Process ESC and CB datasets
        if args.data_name in ["esc", "cb"]:
            for line in infile:
                # Parse JSON line
                sample = json.loads(line.strip("\n"))
                dial = sample["dialog"]
                state = []

                # Process each turn in the dialog
                for turn in dial:
                    if turn["speaker"] == "sys" and len(state) > 0:
                        # Build dialog context within sequence length limit
                        dial_id = []
                        for s in state[::-1]:
                            if len(dial_id) + len(s) > args.max_seq_length:
                                break
                            dial_id = s[1:] + dial_id
                        # Create source and target IDs
                        source_id = s[:1] + dial_id
                        target_id = act.index(turn["strategy"])
                        # Add to feature lists
                        source_ids.append(source_id[-args.max_seq_length + 1 :])
                        target_ids.append(target_id)
                        # Update length statistics
                        avg_dia_len.append(len(source_id))
                        max_dia_len = max(max_dia_len, len(source_id))
                    # Encode current turn and add to state
                    state.append(
                        tokenizer.encode(
                            "%s: %s"
                            % (role_map[args.data_name][turn["speaker"]], turn["text"])
                        )
                    )
        # Process CIMA dataset
        elif args.data_name == "cima":
            for line in infile:
                # Parse line using eval
                sample = eval(line.strip("\n"))
                dial = sample["dialog"]
                state = []

                # Get target strategy ID
                target_id = act.index(sample["strategy"])
                dial_id = []
                # Process each turn in dialog
                for s in dial:
                    s = tokenizer.encode(
                        "%s: %s" % (role_map[args.data_name][s["speaker"]], s["text"])
                    )
                    dial_id += s[1:]
                # Create source ID and add to feature lists
                source_id = s[:1] + dial_id
                source_ids.append(source_id[-args.max_seq_length + 1 :])
                target_ids.append(target_id)
                # Update length statistics
                avg_dia_len.append(len(source_id))
                max_dia_len = max(max_dia_len, len(source_id))

        # Print dataset statistics
        print(
            "{} set, max_dia_len: {}, avg_dia_len: {}".format(
                mode, max_dia_len, float(sum(avg_dia_len)) / len(avg_dia_len)
            )
        )

    # Return processed features
    return {"source_ids": source_ids, "target_ids": target_ids}
