from tkinter import Tk, filedialog, messagebox


def show_message(message):
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message=message)
    root.destroy()


def select_file_in_explorer(init_dir='./', message='Select an EEG file!', file_type="Brain Product", ext=".vhdr"):
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message=message)
    extension = ext if ext[0] == '.' else '.' + ext
    filename = filedialog.askopenfilename(title='Select file',
                                          initialdir=init_dir,
                                          filetypes=((file_type, "*{}".format(extension)), ("all files", "*.*")))
    root.destroy()
    assert len(filename) > 0, 'No file were selected...'
    return filename


def select_base_dir():
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message='Select base directory, which contains all the database folders:\n'
                                             '- Cybathlon_pilot\n'
                                             '- physionet.org\n'
                                             '- TTK')
    base_dir = filedialog.askdirectory(title='Select main database directory')

    root.destroy()
    assert len(base_dir) > 0, 'Base directory is not selected. Cannot run program!'
    return base_dir


if __name__ == '__main__':
    base_dir = select_base_dir()
    print(select_file_in_explorer(base_dir))
