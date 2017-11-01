from macarico.util import make_sequence_mod_data


def main():
    num_ex = 1000
    ex_len = 1
    n_types = 6
    n_labels = 4
    data = make_sequence_mod_data(num_ex, ex_len, n_types, n_labels)
    data = map(lambda x: (x[0][0], x[1][0]), data)
    print(data)

if __name__ == '__main__':
    main()
