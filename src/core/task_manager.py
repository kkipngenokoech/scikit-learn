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
    
    def create_task(self, title: str, description: str = "", list_name: str = None) -> Optional[Task]:
        target_list = self.current_list
        if list_name:
            target_list = self.get_task_list(list_name)
        
        if not target_list:
            return None
        
        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            description=description
        )
        target_list.add_task(task)
        return task
    
    def get_task(self, task_id: str, list_name: str = None) -> Optional[Task]:
        if list_name:
            task_list = self.get_task_list(list_name)
            if task_list:
                return task_list.get_task(task_id)
        else:
            for task_list in self.task_lists:
                task = task_list.get_task(task_id)
                if task:
                    return task
        return None
    
    def complete_task(self, task_id: str, list_name: str = None) -> bool:
        task = self.get_task(task_id, list_name)
        if task:
            task.mark_completed()
            return True
        return False
    
    def uncomplete_task(self, task_id: str, list_name: str = None) -> bool:
        task = self.get_task(task_id, list_name)
        if task:
            task.mark_incomplete()
            return True
        return False
    
    def delete_task(self, task_id: str, list_name: str = None) -> bool:
        if list_name:
            task_list = self.get_task_list(list_name)
            if task_list:
                return task_list.remove_task(task_id)
        else:
            for task_list in self.task_lists:
                if task_list.remove_task(task_id):
                    return True
        return False
    
    def get_all_tasks(self, list_name: str = None) -> List[Task]:
        if list_name:
            task_list = self.get_task_list(list_name)
            return task_list.tasks if task_list else []
        
        all_tasks = []
        for task_list in self.task_lists:
            all_tasks.extend(task_list.tasks)
        return all_tasks
    
    def get_completed_tasks(self, list_name: str = None) -> List[Task]:
        if list_name:
            task_list = self.get_task_list(list_name)
            return task_list.get_completed_tasks() if task_list else []
        
        completed_tasks = []
        for task_list in self.task_lists:
            completed_tasks.extend(task_list.get_completed_tasks())
        return completed_tasks
    
    def get_incomplete_tasks(self, list_name: str = None) -> List[Task]:
        if list_name:
            task_list = self.get_task_list(list_name)
            return task_list.get_incomplete_tasks() if task_list else []
        
        incomplete_tasks = []
        for task_list in self.task_lists:
            incomplete_tasks.extend(task_list.get_incomplete_tasks())
        return incomplete_tasks
