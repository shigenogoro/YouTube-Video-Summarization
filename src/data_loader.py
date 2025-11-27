from datasets import load_dataset

def load_meetingbank():
    """
    Load the MeetingBank dataset from HuggingFace.
    
    The expected dataset ID is "huuuyeah/meetingbank".

    Returns:
        train_data (Dataset): The training split.
        val_data (Dataset): The validation split.
        test_data (Dataset): The testing split.
    """
    try:
        # The correct and working dataset ID
        meetingbank = load_dataset("huuuyeah/meetingbank")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Re-raise or handle the error as needed
        raise

    train_data = meetingbank['train']
    val_data = meetingbank['validation']
    test_data = meetingbank['test']

    # Returns three dataset splits
    return train_data, val_data, test_data

