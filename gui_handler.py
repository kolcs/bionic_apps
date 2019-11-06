from tkinter import Tk, filedialog, messagebox


def show_message(message):
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message=message)
    del root


def select_eeg_file_in_explorer(init_dir):
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message='Select an EEG file!')
    filename = filedialog.askopenfilename(title='Select EEG source file',
                                          initialdir=init_dir,
                                          filetypes=(("Brain Product", "*.vhdr"), ("all files", "*.*")))
    del root
    assert len(filename) > 0, 'No EEG files were selected...'
    return filename


def select_base_dir():
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message='Select base directory, which contains all the database folders:\n'
                                             '- Cybathlon_pilot\n'
                                             '- physionet.org\n'
                                             '- TTK')
    base_dir = filedialog.askdirectory(title='Select main database directory')

    del root
    assert len(base_dir) > 0, 'Base directory is not selected. Cannot run program!'
    return base_dir


if __name__ == '__main__':
    base_dir = select_base_dir()
    print(select_eeg_file_in_explorer(base_dir))
