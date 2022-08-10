def get_file_paths(input_path: str):

    supported_filetypes = [".jsonl.zst", ".txt", ".xz", ".tar.gz", ".jsonl"]

    if input_path.is_dir():
        subfiles_by_type = [
            list(Path(input_path).glob(f"*{type}")) for type in supported_filetypes
        ]
        files = [
            sub_file for subfile_group in subfiles_by_type for sub_file in subfile_group
        ]
        assert files, f"No files with supported types found in directory: {input_path}"
    elif input_path.is_file():
        assert (
            str(input_path).endswith(f_type) for f_type in supported_filetypes
        ), f"input filetype must be one of: {supported_filetypes}"
        files = [input_path]
    else:
        raise FileNotFoundError(f"no such file or directory: {input_path=}")

    return [str(file) for file in files]  # convert PosixPaths to strings
