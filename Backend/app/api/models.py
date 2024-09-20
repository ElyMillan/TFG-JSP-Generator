from typing import List
from pydantic import BaseModel, Field

class MainFunctionDataRequest(BaseModel):
    email: str
    full_name: str
    school_name: str
    size: int
    jobs: List[int]
    machines: List[int]
    distributions: List[str]
    speed_scaling: int
    release_due_date: int
    seeds: List[int]
    pickle_file_output: bool
    json_file_output: bool
    dzn_file_output: bool
    taillard_file_output: bool
    single_folder_output: bool
    custom_folder_name: str
    solver: str