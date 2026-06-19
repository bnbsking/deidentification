from deid import main


class TestMainExample:
    def __init__(self):
        self.key = "example"
        main.register(key=self.key, deid_cls_name="ExampleDeid")

    def test_run(self):
        # out = main.run(key=self.key, raw_text="Patient name is John Doe and his phone number is 123-456-7890.")
        # print(out)
        
        # out = main.run(key=self.key, raw_text="Patient name is John Doe and his phone number is 123-456-7890.")
        # print(out)  # cache is fast

        out = main.run(key=self.key, raw_text=open("/app/tests/integration/deid/hr_candidate1.txt", "r").read())
        print(out)

        main.unregister(key=self.key)
        

if __name__ == "__main__":
    TestMainExample().test_run()
