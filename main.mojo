from python import Python


fn main() raises:
    Python.add_to_path(".")
    let parsed_jobs = Python.import_module("jobparser")
    let thing = parsed_jobs.mongo_connect()
