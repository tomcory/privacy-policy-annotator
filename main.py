import crawler

def fetch_ids_in_file(file_path: str):
    # Create an empty list to store the lines
    pkgs = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Strip newline characters and add to the list
            pkgs.append(line.strip())
            pass

    crawler.crawl_list(pkgs)


if __name__ == '__main__':
    file_path = 'pkgs.txt'
    fetch_ids_in_file(file_path)
