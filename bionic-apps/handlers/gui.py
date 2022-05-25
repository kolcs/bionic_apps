from tkinter import Tk, filedialog, messagebox


def show_message(message, title='BCI'):
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title=title, message=message)
    root.destroy()


def select_files_in_explorer(init_dir='./', message='Select an EEG file!', file_type="EEG files",
                             ext=".vhdr;*.edf;*.gdf;*.fif;*.xdf", no_file_error=True):
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title='BCI', message=message)
    extension = ext if ext[0] == '.' else '.' + ext
    filenames = filedialog.askopenfilenames(title='Select file',
                                            initialdir=init_dir,
                                            filetypes=((file_type, "*{}".format(extension)), ("all files", "*.*")))
    root.destroy()
    if no_file_error:
        assert len(filenames) > 0, 'No file were selected...'
    return filenames


def select_folder_in_explorer(message, dialog_title, title='BCI', no_dir_error=True):
    root = Tk()
    root.withdraw()
    messagebox.showinfo(title=title, message=message)
    base_dir = filedialog.askdirectory(title=dialog_title)

    root.destroy()
    if no_dir_error:
        assert len(base_dir) > 0, 'Base directory is not selected. Cannot run program!'
    return base_dir


def select_base_dir():
    base_dir = select_folder_in_explorer(message='Select base directory, which contains all the database folders:\n'
                                                 '- BCI_comp\n'
                                                 '- Cybathlon_pilot\n'
                                                 '- physionet.org\n'
                                                 '- TTK',
                                         dialog_title='Select main database directory')
    return base_dir


if __name__ == '__main__':
    path = select_base_dir()
    print(select_files_in_explorer(path))
