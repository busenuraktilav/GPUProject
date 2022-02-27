
# file_name = input("Enter file name to pre-process: ")
file_name = "sample.gr"

with open(file_name) as f_old:
    lines = f_old.readlines()
    with open('file_name_processed.txt', 'w') as f_new:
        for line in lines:
            chars = line.split(' ')
            if chars[0] == 'p':
                f_new.write(f"{chars[2]} {chars[2]} {chars[3]}")
            elif chars[0] == 'a':
                f_new.write(f"{chars[1]} {chars[2]} {chars[3]}")

 