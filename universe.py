class Universe:
    def __init__(self, elements):
        self.__elts = []
        self.__num = elements  # number of components
        for i in range(elements):
            self.__elts.append({'rank': 0,
                                'size': 1,
                                'p': i})

    def trace(self, x):
        y = x
        while y != self.__elts[y]['p']:
            y = self.__elts[y]['p']

        self.__elts[x]['p'] = y
        return y

    def join(self, x, y):
        if self.__elts[x]['rank'] > self.__elts[y]['rank']:
            self.__elts[y]['p'] = x
            self.__elts[x]['size'] += self.__elts[y]['size']
        else:
            self.__elts[x]['p'] = y
            self.__elts[y]['size'] += self.__elts[x]['size']
            if self.__elts[x]['rank'] == self.__elts[y]['rank']:
                self.__elts[y]['rank'] += 1
        self.__num -= 1

    def get_size_of(self, x):
        return self.__elts[x]['size']

    def get_num(self):
        return self.__num