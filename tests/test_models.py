import pytest
from datetime import datetime
from src.core.models import Task, TaskList


class TestTask:
    def test_task_creation(self):
        task = Task(id="1", title="Test Task")
        assert task.id == "1"
        assert task.title == "Test Task"
        assert task.description == ""
        assert task.completed is False
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)
    
    def test_task_with_description(self):
        task = Task(id="1", title="Test Task", description="Test Description")
        assert task.description == "Test Description"
    
    def test_mark_completed(self):
        task = Task(id="1", title="Test Task")
        original_updated_at = task.updated_at
        task.mark_completed()
        assert task.completed is True
        assert task.updated_at > original_updated_at
    
    def test_mark_incomplete(self):
        task = Task(id="1", title="Test Task", completed=True)
        original_updated_at = task.updated_at
        task.mark_incomplete()
        assert task.completed is False
        assert task.updated_at > original_updated_at
    
    def test_update_title(self):
        task = Task(id="1", title="Old Title")
        original_updated_at = task.updated_at
        task.update_title("New Title")
        assert task.title == "New Title"
        assert task.updated_at > original_updated_at
    
    def test_update_description(self):
        task = Task(id="1", title="Test Task")
        original_updated_at = task.updated_at
        task.update_description("New Description")
        assert task.description == "New Description"
        assert task.updated_at > original_updated_at


class TestTaskList:
    def test_task_list_creation(self):
        task_list = TaskList(name="My List")
        assert task_list.name == "My List"
        assert task_list.tasks == []
        assert isinstance(task_list.created_at, datetime)
    
    def test_add_task(self):
        task_list = TaskList(name="My List")
        task = Task(id="1", title="Test Task")
        task_list.add_task(task)
        assert len(task_list.tasks) == 1
        assert task_list.tasks[0] == task
    
    def test_remove_task(self):
        task_list = TaskList(name="My List")
        task = Task(id="1", title="Test Task")
        task_list.add_task(task)
        
        result = task_list.remove_task("1")
        assert result is True
        assert len(task_list.tasks) == 0
        
        result = task_list.remove_task("nonexistent")
        assert result is False
    
    def test_get_task(self):
        task_list = TaskList(name="My List")
        task = Task(id="1", title="Test Task")
        task_list.add_task(task)
        
        found_task = task_list.get_task("1")
        assert found_task == task
        
        not_found = task_list.get_task("nonexistent")
        assert not_found is None
    
    def test_get_completed_tasks(self):
        task_list = TaskList(name="My List")
        task1 = Task(id="1", title="Task 1", completed=True)
        task2 = Task(id="2", title="Task 2", completed=False)
        task3 = Task(id="3", title="Task 3", completed=True)
        
        task_list.add_task(task1)
        task_list.add_task(task2)
        task_list.add_task(task3)
        
        completed_tasks = task_list.get_completed_tasks()
        assert len(completed_tasks) == 2
        assert task1 in completed_tasks
        assert task3 in completed_tasks
        assert task2 not in completed_tasks
    
    def test_get_incomplete_tasks(self):
        task_list = TaskList(name="My List")
        task1 = Task(id="1", title="Task 1", completed=True)
        task2 = Task(id="2", title="Task 2", completed=False)
        task3 = Task(id="3", title="Task 3", completed=False)
        
        task_list.add_task(task1)
        task_list.add_task(task2)
        task_list.add_task(task3)
        
        incomplete_tasks = task_list.get_incomplete_tasks()
        assert len(incomplete_tasks) == 2
        assert task2 in incomplete_tasks
        assert task3 in incomplete_tasks
        assert task1 not in incomplete_tasks
