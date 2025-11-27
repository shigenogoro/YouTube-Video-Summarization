from datasets import load_dataset

def load_meetingbank():
    """
    Load the MeetingBank dataset from HuggingFace.
    Returns:
        train_data: Dataset
        test_data: Dataset
    """
    dataset = load_dataset("MeetingBank-HF/MeetingBank")
    return dataset["train"], dataset["test"]
