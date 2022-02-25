import os 

class Filetools:

    @classmethod
    def save_to_file(cls, filename, target_list, verbose=True):
        """ Saves a specified list to a .mem file in the cache with entry comments.
        
        Parameters
        ----------
        filename: str
            The desired filename.
        target_list: [str]
            The binary parameters supplied as a list of strings.
        """

        path = cls.open_cache(filename)
        print(path)
        file = open(path, "w")

        for i, element in enumerate(target_list):
            if verbose:
                file.write("//" + "Entry " + str(i) + '\n')
            file.write(element + " " + '\n')

    @staticmethod
    def open_cache(filename):
        # Obtain path to cache
        path = os.path.realpath(__file__)
        path = path.replace("/tools", "")
        dir = os.path.dirname(path) + "/cache"

        # Create the cache if it does not exist
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        return str(dir) + "/" + filename + ".mem"