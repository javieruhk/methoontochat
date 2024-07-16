import os
from grobid_client.grobid_client import GrobidClient, ServerUnavailableException
import xml.etree.ElementTree as ET

#docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.0

try:
    input_path="./data/input/17 theoretical files + 9 practical files"
    output="./data/output/xml files"
    client = GrobidClient(config_path='./config.json')
    
    client.process(
        service="processFulltextDocument", 
        input_path=input_path, 
        output=output, 
        verbose=True
    )

    def extract_body_from_xml(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        body_node = root.find('.//{http://www.tei-c.org/ns/1.0}body')

        if body_node is not None:
            body_content = ""
            for child in body_node.iter():
                if child.tag.endswith('head'):
                    if child.text:
                        body_content += child.text + "\n"
                elif child.tag.endswith('p'):
                    if child.text:
                        body_content += child.text
            return body_content.strip()

        else:
            print(f"No se encontró el body en el archivo: {xml_file}")
            return ""
    
    for file in os.listdir(output):
        if file.endswith('.xml'):
            xml_filename = os.path.basename(file)
            filename = xml_filename.replace(".grobid.tei.xml", "")
            body = extract_body_from_xml('./data/output/xml files/' + xml_filename)
            output_file_path = './data/output/txt files/'+ filename + '.txt'

            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(body)

except ServerUnavailableException:
    print("El servidor de GROBID no está en ejecución, la conexión ha fallado")