def test_main_imports_answer_question_workflow():
    import main

    assert callable(main.answer_question)
