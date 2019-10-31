from tkinter import Tk, filedialog, messagebox


def select_eeg_file_in_explorer():
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message='Select an EEG file for training BCI System!')
    filename = filedialog.askopenfilename(title='Select EEG source file',
                                          filetypes=(("Brain Product", "*.vhdr"), ("all files", "*.*")))
    del root
    return filename if len(filename) > 0 else None


def select_base_dir():
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message='Select base dir, which contains all the databases: \n'
                                             '- Cybathlon_pilot\n'
                                             '- physionet.org\n'
                                             '- TTK')
    base_dir = filedialog.askdirectory(title='Select main database directory')

    del root
    return base_dir if len(base_dir) > 0 else None


if __name__ == '__main__':
    print(select_eeg_file_in_explorer())
    print(select_base_dir())
