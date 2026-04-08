from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    completed: bool = False
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def mark_completed(self):
        self.completed = True
        self.updated_at = datetime.now()
    
    def mark_incomplete(self):
        self.completed = False
        self.updated_at = datetime.now()
    
    def update_title(self, title: str):
        self.title = title
        self.updated_at = datetime.now()
    
    def update_description(self, description: str):
        self.description = description
        self.updated_at = datetime.now()


@dataclass
class TaskList:
    name: str
    tasks: List[Task] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def add_task(self, task: Task):
        self.tasks.append(task)
    
    def remove_task(self, task_id: str) -> bool:
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                del self.tasks[i]
                return True
        return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_completed_tasks(self) -> List[Task]:
        return [task for task in self.tasks if task.completed]
    
    def get_incomplete_tasks(self) -> List[Task]:
        return [task for task in self.tasks if not task.completed]
