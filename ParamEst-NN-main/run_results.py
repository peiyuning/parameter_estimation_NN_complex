from multiprocessing import freeze_support

if __name__ == "__main__":
#    freeze_support()

    import runpy
    runpy.run_path("notebooks/3-Results.py", run_name="__main__")
