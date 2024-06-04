import os
from grobid_client.grobid_client import GrobidClient, ServerUnavailableException

import xml.etree.ElementTree as ET

#docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.0

try:
    input_path="./data/pdf files"
    output="./data/xml files"

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
            body_content = ET.tostring(body_node, encoding='utf-8', method='text').decode('utf-8').strip()
            return body_content
        else:
            print(f"No se encontró el body en el archivo: {xml_file}")
            return ""
    
    for file in os.listdir(output):
        if file.endswith('.xml'):
            xml_filename = os.path.basename(file)
            filename = xml_filename.replace(".grobid.tei.xml", "")

            print(xml_filename)
            print(filename)

            body = extract_body_from_xml('./data/xml files/' + xml_filename)

            #print(body)


            output_file_path = './data/txt files/'+ filename + '.txt'

            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(body)

except ServerUnavailableException:
    print("El servidor de GROBID no está en ejecución, la conexión ha fallado")

    

