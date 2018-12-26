# -*- coding: utf-8 -*-
import numpy as np
import os
import re
import operator
import pickle
import sys


class Dataset:

    def __init__(self, parameter, extern_data):
        self.parameter = parameter
        self.extern_data = extern_data

        if parameter["mode"] == "train" and not os.path.exists(parameter["necessary_file"]):
            self._make_necessary_data_by_train_data()
        else:
            with open(parameter["necessary_file"], 'rb') as f:
                self.necessary_data = pickle.load(f)

        self.parameter["embedding"] = [
            ["word", len(self.necessary_data["word"]), parameter["word_embedding_size"]],
            ["character", len(self.necessary_data["character"]), parameter["char_embedding_size"]]
        ]

        self.parameter["n_class"] = len(self.necessary_data["ner_tag"])

        self.tr_morphs, self.te_morphs = None, None
        self.tr_ne_dicts, self.te_ne_dicts = None, None
        self.tr_characters, self.te_characters = None, None
        self.tr_sequence_lengths, self.te_sequence_lengths = None, None
        self.tr_character_lengths, self.te_character_lengths = None, None
        self.tr_labels, self.te_labels = None, None

        np.random.seed(1337)

    def _make_necessary_data_by_train_data(self):
        necessary_data = {"word": {}, "character": {}, "ner_tag": {}, "ner_morph_tag": {}}

        for morphs, tags, ner_tag, ner_mor_list, ner_tag_list in self._read_data_file(extern_data=self.extern_data):
            for mor, tag in zip(morphs, tags):
                self._check_dictionary(necessary_data["word"], mor)

                for char in mor:
                    self._check_dictionary(necessary_data["character"], char)

            if type(ner_tag) is list:
                for ne in ner_tag:
                    if ne == "-":
                        continue
                    self._check_dictionary(necessary_data["ner_tag"], ne + "_B")
                    self._check_dictionary(necessary_data["ner_tag"], ne + "_I")
            else:
                self._check_dictionary(necessary_data["ner_tag"], ner_tag + "_B")
                self._check_dictionary(necessary_data["ner_tag"], ner_tag + "_I")

            for nerMor, nerTag in zip(ner_mor_list, ner_tag_list):
                if nerTag == "-" or nerTag == "-_B":
                    continue
                nerTag = nerTag.split("_")[0]
                self._check_dictionary(necessary_data["ner_morph_tag"], nerMor, nerTag)

        # 존재하는 어절 사전
        necessary_data["word"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["word"], start=2)

        # 존재하는 음절 사전
        necessary_data["character"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["character"],
                                                                                    start=2)

        # 존재하는 NER 품사 태그 사전
        necessary_data["ner_tag"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["ner_tag"], start=2,
                                                                                  unk=False)
        self.ner_tag_size = len(necessary_data["ner_tag"])
        self.necessary_data = necessary_data

        # 존재하는 형태소 별 NER 품사 태그 비율 사전
        necessary_data["ner_morph_tag"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["ner_morph_tag"],
                                                                                        start=0, ner=True)

        with open(self.parameter["necessary_file"], 'wb') as f:
            pickle.dump(necessary_data, f)

    def make_input_data(self, extern_data=None):
        morphs = []
        ne_dicts = []
        characters = []
        labels = []
        sequence_lengths = []
        character_lengths = []

        if extern_data is not None:
            self.extern_data = extern_data

        temp = [[], [], []]
        # TAG 정보가 없는 경우에는 tag 자리에 mor 정보가 들어온다
        for mor, tag, _, ner_mor, ner_tag in self._read_data_file(pre=False, extern_data=self.extern_data):
            if tag != False:
                temp[0] += mor
                temp[1] += tag
                if len(ner_tag) == 0:
                    temp[2] += ['O'] * len(mor)
                elif len(ner_tag) == len(mor):
                    temp[2] = ner_tag
                else:
                    for i, m in enumerate(mor):
                        if m == ner_mor[0]:
                            break
                    ner_tag = ['O'] * i + ner_tag
                    ner_tag = ner_tag + ['O'] * (len(mor) - len(ner_tag))
                    temp[2] += ner_tag
            else:
                morph = [0] * self.parameter["sentence_length"]
                ne_dict = [[0.] * int(self.parameter["n_class"] / 2)] * self.parameter["sentence_length"]
                character = [[0] * self.parameter["word_length"]] * self.parameter["sentence_length"]
                character_length = [0] * self.parameter["sentence_length"]
                label = [0] * self.parameter["sentence_length"]

                if len(temp[0]) > self.parameter["sentence_length"]:
                    temp = [[], [], []]
                    continue

                sequence_lengths.append(len(temp[0]))
                for mor, tag, neTag, index in zip(temp[0], temp[1], temp[2], range(0, len(temp[0]))):
                    morph[index] = self._search_index_by_dict(self.necessary_data["word"], mor)
                    ne_dict[index] = self._search_index_by_dict(self.necessary_data["ner_morph_tag"], mor)
                    if neTag != "-" and neTag != "-_B":
                        label[index] = self._search_index_by_dict(self.necessary_data["ner_tag"], neTag)
                    sub_char = [0] * self.parameter["word_length"]
                    for i, char in enumerate(mor):
                        if i == self.parameter["word_length"]:
                            i -= 1
                            break
                        sub_char[i] = self._search_index_by_dict(self.necessary_data["character"], char)
                    character_length[index] = i + 1
                    character[index] = sub_char

                morphs.append(morph)
                ne_dicts.append(ne_dict)
                characters.append(character)
                character_lengths.append(character_length)
                labels.append(label)

                temp = [[], [], []]

        self.morphs = np.array(morphs)
        self.ne_dicts = np.array(ne_dicts)
        self.characters = np.array(characters)
        self.sequence_lengths = np.array(sequence_lengths)
        self.character_lengths = np.array(character_lengths)
        self.labels = np.array(labels)

        print("[*] morphs      : ", self.morphs.shape)
        print("[*] ne_dicts    : ", self.ne_dicts.shape)
        print("[*] characters  : ", self.characters.shape)
        print("[*] seq_length  : ", self.sequence_lengths.shape)
        print("[*] char_length : ", self.character_lengths.shape)
        print("[*] label       : ", self.labels.shape)

        print("[*] Example :")
        print("[*] morphs      : ", self.morphs[0])
        print("[*] morphs mask : ", np.sign(np.abs(self.morphs[0])))
        print("[*] ne_dicts    : ", self.ne_dicts[0])
        print("[*] characters  : ", self.characters[0])
        print("[*] seq_length  : ", self.sequence_lengths[0])
        print("[*] char_length : ", self.character_lengths[0])
        print("[*] label       : ", self.labels[0])

        if self.parameter["mode"] == "train":  # 90% / 10%
            tr_size = 900
            self.tr_morphs, self.te_morphs = self.morphs[:tr_size, :], self.morphs[tr_size:, :]
            self.tr_ne_dicts, self.te_ne_dicts = self.ne_dicts[:tr_size, :], self.ne_dicts[tr_size:, :]
            self.tr_characters, self.te_characters = self.characters[:tr_size, :], self.characters[tr_size:, :]
            self.tr_sequence_lengths, self.te_sequence_lengths = \
                self.sequence_lengths[:tr_size], self.sequence_lengths[tr_size:]
            self.tr_character_lengths, self.te_character_lengths = \
                self.character_lengths[:tr_size, :], self.character_lengths[tr_size:, :]
            self.tr_labels, self.te_labels = self.labels[:tr_size, :], self.labels[tr_size:, :]

            print("[+] split into train/valid")
            print("[*] morphs      : ", self.tr_morphs.shape, self.te_morphs.shape)
            print("[*] ne_dicts    : ", self.tr_ne_dicts.shape, self.te_ne_dicts.shape)
            print("[*] characters  : ", self.tr_characters.shape, self.te_characters.shape)
            print("[*] seq_length  : ", self.tr_sequence_lengths.shape, self.te_sequence_lengths.shape)
            print("[*] char_length : ", self.tr_character_lengths.shape, self.te_character_lengths.shape)
            print("[*] label       : ", self.tr_labels.shape, self.te_labels.shape)

    def shuffle_data(self):
        s = np.arange(self.tr_morphs.shape[0])
        np.random.shuffle(s)

        self.tr_morphs = self.tr_morphs[s]
        self.tr_ne_dicts = self.tr_ne_dicts[s]
        self.tr_characters = self.tr_characters[s]
        self.tr_sequence_lengths = self.tr_sequence_lengths[s]
        self.tr_character_lengths = self.tr_character_lengths[s]
        self.tr_labels = self.tr_labels[s]

    def get_data_batch_size(self, n, train=True, valid=False):
        if train:
            if valid:
                for i, step in enumerate(range(0, self.parameter["train_lines"], n)):
                    if len(self.te_morphs[step:step + n]) == n:
                        yield self.te_morphs[step:step + n], self.te_ne_dicts[step:step + n], \
                              self.te_characters[step:step + n], self.te_sequence_lengths[step:step + n], \
                              self.te_character_lengths[step:step + n], self.te_labels[step:step + n], i
            else:
                for i, step in enumerate(range(0, self.parameter["train_lines"], n)):
                    if len(self.tr_morphs[step:step + n]) == n:
                        yield self.tr_morphs[step:step + n], self.tr_ne_dicts[step:step + n],\
                              self.tr_characters[step:step + n], self.tr_sequence_lengths[step:step + n],\
                              self.tr_character_lengths[step:step + n], self.tr_labels[step:step + n], i
        else:
            for i, step in enumerate(range(0, self.parameter["train_lines"], n)):
                if len(self.morphs[step:step + n]) == n:
                    yield self.morphs[step:step + n], self.ne_dicts[step:step + n],\
                          self.characters[step:step + n], self.sequence_lengths[step:step + n],\
                          self.character_lengths[step:step + n], self.labels[step:step + n], i

    def get_valid_data(self):
        return self.te_morphs, self.te_ne_dicts, self.te_characters, self.te_sequence_lengths, \
               self.te_character_lengths, self.te_labels

    def _search_index_by_dict(self, dict, key):
        if key in dict:
            return dict[key]
        else:
            if "UNK" in dict:
                return dict["UNK"]
            else:
                temp = [0.0] * int(self.parameter["n_class"] / 2)
                temp[0] = 1.0
                return temp

    def _read_data_file(self, pre=True, extern_data=None):
        if extern_data is not None:
            return self._read_extern_data_file(pre, self.extern_data)

    def _read_extern_data_file(self, pre=True, extern_data=None):
        cntLine = 0
        for sentence in extern_data:
            morphs = []
            tags = []
            ner_tag = []
            ner_mor_list = []
            for morph in sentence[1]:
                morphs.append(morph)
                tags.append(morph)
                ner_mor_list.append(morph)
            seq_len = len(morphs)

            ner_tag_list = ['O'] * seq_len
            for index, ne in enumerate(sentence[2]):
                ner_tag.append(ne.split("_")[0])
                ner_tag_list[index] = ne

            yield morphs, tags, ner_tag, ner_mor_list, ner_tag_list
            cntLine += 1
            if pre == False:
                yield [], False, False, False, False
            if cntLine % 1000 == 0:
                sys.stderr.write("%d Lines .... \r" % cntLine)

                if self.parameter["train_lines"] < cntLine:
                    break

    @staticmethod
    def _check_dictionary(dict, data, value=0):
        if type(value) is int:
            if not data in dict:
                dict[data] = value
        elif type(value) is str:
            if not value in dict:
                dict[data] = {value: 1}
            else:
                if value in dict[data]:
                    dict[data][value] += 1
                else:
                    dict[data][value] = 1

    def _necessary_data_sorting_and_reverse_dict(self, dict, start=1, unk=True, ner=False):
        dict_temp = {}
        index = start

        if start == 2:
            dict_temp["PAD"] = 0
            if unk:
                dict_temp["UNK"] = 1
            else:
                dict_temp["O"] = 1
        elif start == 1:
            dict_temp["PAD"] = 0

        for key in sorted(dict.items(), key=operator.itemgetter(0), reverse=False):
            if ner:
                items = np.zeros(int(self.ner_tag_size / 2))
                for i in key[1]:
                    items[int(self.necessary_data["ner_tag"][i + "_B"] / 2)] = dict[key[0]][i]
                dict_temp[key[0]] = items / np.sum(items)
            else:
                dict_temp[key[0]] = index
                index += 1

        return dict_temp


if __name__ == "__main__":
    dataset = Dataset({"input_dir": "data/NER.sample.txt"})
