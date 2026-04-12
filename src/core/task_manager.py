from typing import List, Optional
import uuid
from .models import Task, TaskList


class TaskManager:
    def __init__(self):
        self.task_lists: List[TaskList] = []
        self.current_list: Optional[TaskList] = None
    
    def create_task_list(self, name: str) -> TaskList:
        task_list = TaskList(name=name)
        self.task_lists.append(task_list)
        if self.current_list is None:
            self.current_list = task_list
        return task_list
    
    def get_task_list(self, name: str) -> Optional[TaskList]:
        for task_list in self.task_lists:
            if task_list.name == name:
                return task_list
        return None
    
    def set_current_list(self, name: str) -> bool:
        task_list = self.get_task_list(name)
        if task_list:
            self.current_list = task_list
            return True
        return False
    
    def create_task(self, title: str, description: str = "") -> Optional[Task]:
        if not self.current_list:
            return None
        
        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            description=description
        )
        self.current_list.add_task(task)
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        if not self.current_list:
            return None
        return self.current_list.get_task(task_id)
    
    def update_task(self, task_id: str, title: str = None, description: str = None) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        
        if title is not None:
            task.update_title(title)
        if description is not None:
            task.update_description(description)
        return True
    
    def complete_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        task.mark_completed()
        return True
    
    def uncomplete_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        task.mark_incomplete()
        return True
    
    def delete_task(self, task_id: str) -> bool:
        if not self.current_list:
            return False
        return self.current_list.remove_task(task_id)
    
    def list_tasks(self, completed: Optional[bool] = None) -> List[Task]:
        if not self.current_list:
            return []
        
        if completed is None:
            return self.current_list.tasks
        elif completed:
            return self.current_list.get_completed_tasks()
        else:
            return self.current_list.get_incomplete_tasks()
