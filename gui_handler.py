from tkinter import Tk, filedialog, messagebox


def select_eeg_file_in_explorer():
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message='Select an EEG file!')
    filename = filedialog.askopenfilename(title='Select EEG source file',
                                          filetypes=(("Brain Product", "*.vhdr"), ("all files", "*.*")))
    del root
    assert len(filename) > 0, 'No source files were selected...'
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
    print(select_eeg_file_in_explorer())
    print(select_base_dir())
