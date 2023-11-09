#!/usr/bin/python3

def test_func(nb: int, pos: int):
    tab = [0 if i != pos else 1 for i in range(nb)]
    print(tab)

if __name__ == "__main__":
    test_func(2, 1)
    test_func(3, 1)
    test_func(4, 2)
    test_func(5, 2)
