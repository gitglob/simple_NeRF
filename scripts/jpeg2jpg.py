import os

def rename_files(directory):
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.JPEG'):
                # Construct full file path
                old_file_path = os.path.join(root, file)
                # Construct new file path with .JPG extension
                new_file_path = os.path.join(root, file[:-5] + '.JPG')
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} to {new_file_path}')

def main():
    # Specify the directory
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(current_file_path, "..", "data", "custom")
    print(directory_path)
    rename_files(directory_path)


if __name__ == "__main__":
    main()
