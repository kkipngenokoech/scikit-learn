import pytest
from src.core.task_manager import TaskManager
from src.core.models import Task, TaskList


class TestTaskManager:
    def test_create_task_list(self):
        manager = TaskManager()
        task_list = manager.create_task_list("Work")
        
        assert task_list.name == "Work"
        assert len(manager.task_lists) == 1
        assert manager.current_list == task_list
    
    def test_get_task_list(self):
        manager = TaskManager()
        task_list = manager.create_task_list("Work")
        
        found_list = manager.get_task_list("Work")
        assert found_list == task_list
        
        not_found = manager.get_task_list("Nonexistent")
        assert not_found is None
    
    def test_set_current_list(self):
        manager = TaskManager()
        list1 = manager.create_task_list("Work")
        list2 = manager.create_task_list("Personal")
        
        assert manager.current_list == list1
        
        result = manager.set_current_list("Personal")
        assert result is True
        assert manager.current_list == list2
        
        result = manager.set_current_list("Nonexistent")
        assert result is False
        assert manager.current_list == list2
    
    def test_create_task(self):
        manager = TaskManager()
        manager.create_task_list("Work")
        
        task = manager.create_task("Test Task", "Test Description")
        assert task is not None
        assert task.title == "Test Task"
        assert task.description == "Test Description"
        assert len(manager.current_list.tasks) == 1
    
    def test_create_task_no_current_list(self):
        manager = TaskManager()
        task = manager.create_task("Test Task")
        assert task is None
    
    def test_get_task(self):
        manager = TaskManager()
        manager.create_task_list("Work")
        task = manager.create_task("Test Task")
        
        found_task = manager.get_task(task.id)
        assert found_task == task
        
        not_found = manager.get_task("nonexistent")
        assert not_found is None
    
    def test_update_task(self):
        manager = TaskManager()
        manager.create_task_list("Work")
        task = manager.create_task("Old Title", "Old Description")
        
        result = manager.update_task(task.id, title="New Title")
        assert result is True
        assert task.title == "New Title"
        assert task.description == "Old Description"
        
        result = manager.update_task(task.id, description="New Description")
        assert result is True
        assert task.description == "New Description"
        
        result = manager.update_task("nonexistent", title="Test")
        assert result is False
    
    def test_complete_task(self):
        manager = TaskManager()
        manager.create_task_list("Work")
        task = manager.create_task("Test Task")
        
        result = manager.complete_task(task.id)
        assert result is True
        assert task.completed is True
        
        result = manager.complete_task("nonexistent")
        assert result is False
    
    def test_uncomplete_task(self):
        manager = TaskManager()
        manager.create_task_list("Work")
        task = manager.create_task("Test Task")
        task.mark_completed()
        
        result = manager.uncomplete_task(task.id)
        assert result is True
        assert task.completed is False
        
        result = manager.uncomplete_task("nonexistent")
        assert result is False
    
    def test_delete_task(self):
        manager = TaskManager()
        manager.create_task_list("Work")
        task = manager.create_task("Test Task")
        
        result = manager.delete_task(task.id)
        assert result is True
        assert len(manager.current_list.tasks) == 0
        
        result = manager.delete_task("nonexistent")
        assert result is False
    
    def test_list_tasks(self):
        manager = TaskManager()
        manager.create_task_list("Work")
        
        task1 = manager.create_task("Task 1")
        task2 = manager.create_task("Task 2")
        task3 = manager.create_task("Task 3")
        
        task1.mark_completed()
        task3.mark_completed()
        
        all_tasks = manager.list_tasks()
        assert len(all_tasks) == 3
        
        completed_tasks = manager.list_tasks(completed=True)
        assert len(completed_tasks) == 2
        assert task1 in completed_tasks
        assert task3 in completed_tasks
        
        incomplete_tasks = manager.list_tasks(completed=False)
        assert len(incomplete_tasks) == 1
        assert task2 in incomplete_tasks
    
    def test_list_tasks_no_current_list(self):
        manager = TaskManager()
        tasks = manager.list_tasks()
        assert tasks == []
