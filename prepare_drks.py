import os

from lxml import etree
from lxml.etree import XMLParser
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from utils import LogUtil, DataHandler

icd10_chapter_mappings = {
    "A00-B99": "I",
    "C00-D48": "II",
    "D50-D90": "III",
    "E00-E90": "IV",
    "F00-F99": "V",
    "G00-G99": "VI",
    "H00-H59": "VII",
    "H60-H95": "VIII",
    "I00-I99": "IX",
    "J00-J99": "X",
    "K00-K93": "XI",
    "L00-L99": "XII",
    "M00-M99": "XIII",
    "N00-N99": "XIV",
    "O00-O99": "XV",
    "P00-P96": "XVI",
    "Q00-Q99": "XVII",
    "R00-R99": "XVIII",
    "S00-T98": "XIX",
    "V00-Y84": "XX",
    "Z00-Z99": "XXI",
    "U00-U99": "XXII"
}

class DrksParser(object):

    def __init__(self):
        self.logger = LogUtil.get_logger(self.__class__.__name__, "_logs")
        self.xml_parser = XMLParser(encoding="utf-8", huge_tree=True, ns_clean=True)

    def build_data_set(self, root_dir: str):
        entries = self.parse_folder(root_dir)

        data_handler = DataHandler()
        icd10_ontology = data_handler.read_icd10_ontology()
        group_codes = set(data_handler.read_reduced_icd10_ontology().keys())

        de_entries = dict()
        en_entries = dict()
        for entry in entries:
            if "de_" + entry[0] in de_entries:
                continue

            icd10_codes = entry[3]

            valid_group_codes = set()
            main_chapter = ""

            for code in icd10_codes:
                if code in icd10_chapter_mappings:
                    code = icd10_chapter_mappings[code]

                if not code in icd10_ontology and "." in code:
                    code = code[:code.index(".")]

                if not code in icd10_ontology:
                    self.logger.error(f"Can't find code {code} in ICD10 ontology")
                    continue

                path_components = icd10_ontology[code].split("#")
                if main_chapter == "":
                    main_chapter = path_components[0]

                for path_comp in path_components:
                    if path_comp in group_codes:
                        valid_group_codes.add(path_comp)

            valid_group_codes = "|".join(valid_group_codes)

            de_entries["de_" + entry[0]] = {"text": entry[1], "language": "de", "all_labels": valid_group_codes, "main_chapter": main_chapter}
            en_entries["en_" + entry[0]] = {"text": entry[2], "language": "en", "all_labels": valid_group_codes, "main_chapter": main_chapter}

        de_df = DataFrame.from_dict(de_entries, orient="index")
        en_df = DataFrame.from_dict(en_entries, orient="index")

        de_train, de_dev = train_test_split(de_df, train_size=0.8, stratify=de_df["main_chapter"])
        de_train["data_set"] = "train"
        de_dev["data_set"] = "dev"
        de_df = de_train.append(de_dev)

        en_train, en_dev = train_test_split(en_df, train_size=0.8, stratify=en_df["main_chapter"])
        en_train["data_set"] = "train"
        en_dev["data_set"] = "dev"
        en_df = en_train.append(en_dev)

        full_df = de_df.append(en_df)
        full_df = full_df.drop_duplicates()
        full_df = full_df[full_df["text"].notna()]

        #de_df.to_csv("drks_de.tsv", sep="\t", columns=["data_set", "main_chapter", "all_labels", "text"], index_label="id")
        #en_df.to_csv("drks_en.tsv", sep="\t", columns=["data_set", "main_chapter", "all_labels", "text"], index_label="id")

        output_dir = "data/drks/prepared"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "drks_full.tsv")
        full_df.to_csv(output_file, sep="\t", columns=["language", "data_set", "main_chapter", "all_labels", "text"], index_label="id")

    def parse_folder(self, directory: str):
        entries = []
        for file in os.listdir(directory):
            path = os.path.join(directory, file)

            if os.path.isfile(path) and path.endswith(".xml"):
                entries.append(self.parse_file(path))
            elif os.path.isdir(path):
                for entry in self.parse_folder(path):
                    entries.append(entry)

        return entries

    def parse_file(self, file: str):
        tree = etree.parse(file, self.xml_parser)
        root = tree.getroot()

        id = root.findall("{urn::trial}drksId")[0].text

        en_text = ""
        de_text = ""

        title_element = root.findall(".//{urn::trial}title")[0]
        for content in title_element.findall(".//{urn::trial}localizedContent"):
            locale = content.attrib["locale"]
            text = content.text if content.text is not None else ""

            if locale == "de":
                de_text += text.replace("\t", "").replace("\n", "") + " "
            elif locale == "en":
                en_text += text.replace("\t", "").replace("\n", "") + " "
            else:
                self.logger.error(f"Unknown locale {locale}")

        synposis_element = root.findall(".//{urn::trial}scientificSynopsis")[0]
        for content in synposis_element.findall(".//{urn::trial}localizedContent"):
            locale = content.attrib["locale"]
            text = content.text if content.text is not None else ""

            if locale == "de":
                de_text += text.replace("\t", "").replace("\n", "") + " "
            elif locale == "en":
                en_text += text.replace("\t", "").replace("\n", "") + " "
            else:
                self.logger.error(f"Unknown locale {locale}")

        summary_element = root.findall(".//{urn::trial}publicSummary")[0]
        for content in summary_element.findall(".//{urn::trial}localizedContent"):
            locale = content.attrib["locale"]
            text = content.text if content.text is not None else ""

            if locale == "de":
                de_text += text.replace("\t", "").replace("\n", "") + " "
            elif locale == "en":
                en_text += text.replace("\t", "").replace("\n", "") + " "
            else:
                self.logger.error(f"Unknown locale {locale}")

        de_text = de_text.strip()
        if len(de_text) == 0:
            de_text = None

        en_text = en_text.strip()
        if len(en_text) == 0:
            en_text = None

        icd10_codes = set()
        indication_elements = root.findall(".//{urn::trial}indication")
        for indication in indication_elements:
            # Check type / classification system of indication
            type = indication.findall(".//{urn::trial}type")[0].attrib["key"]
            if type != "icd10":
                #self.logger.info(f"Found unsupported indication type {type}")
                continue

            values = indication.findall(".//{urn::trial}value")
            if len(values) > 0 and "key" in values[0].attrib:
                icd10_code = values[0].attrib["key"]
                icd10_codes.add(icd10_code)


        return id, de_text, en_text, icd10_codes


if __name__ == "__main__":
    parser = DrksParser()
    #result = parser.build_data_set("_out/drks/")
    result = parser.build_data_set("data/drks/raw/")

