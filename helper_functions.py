import os
import os.path

def check_make(dirM):
    """
    Checks to see if the given directory, or list of directories, exists, and if not, creates it.
    :param dirM: Directory, or list of directories
    :return: List of booleans corresponding to the directories - returns "True" if the directory already exists or "False" if it did not exist and was created.
    """

    # Put dirM in list to iterate (allows function to be able to handle single input or multiple inputs)
    if not isinstance(dirM, (list, tuple)):
        dirM = [dirM]

    output_list = []
    for dirm in dirM:
        if not os.path.exists(dirm):
            # The dir does not exist
            os.makedirs(dirm)
            output_list.append(False)
        else:
            # The dir exists
            output_list.append(True)

    return(output_list)
