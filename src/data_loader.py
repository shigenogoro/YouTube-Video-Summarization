from datasets import load_dataset

def load_meetingbank():
    """
    Load the MeetingBank dataset from HuggingFace.
    Returns:
        train_data: Dataset
        val_data: Dataset
        test_data: Dataset
    """
    meetingbank = load_dataset("huuuyeah/meetingbank")

    train_data = meetingbank['train']
    test_data = meetingbank['test']
    val_data = meetingbank['validation']

    return train_data, val_data, test_data

