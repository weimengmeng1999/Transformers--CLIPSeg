import os


seg_class_description_dict = {
    'Background': 'These are body tissues that are not the kidney or the small intestine. There are red blood on its surface, and the size, shape of internal surface can vary. Some parts of organ surface is smooth and white with light reflections.',
    'Instrument': 'Instrument',
    'Shaft': 'The instrument shaft is a cylindrical mettalic instrument used for carrying a device. The end of instrument shaft is usually connected to the Instrument-wrist which carries a device such as the Instrument-clasper.',
    'Wrist': 'The instrument wrist is a connector which connects the end-effector (the device at the end of a robotic arm) to the instrument-shaft. It is made up of different metallic materials acting similarly to the joint of a robotic arm.',
    'Claspers': 'The primary function of instrument-clasper is the manipulation of tissues. To fulfil this role of manipulation, instrument-clasper has a scissor-like appearance. It is one of the end-effectors (the device at the end of a robotic arm) of a robotic surgical system.',
    'Bipolar Forceps': 'Bipolar forceps have a double-action fine curved jaws with horizontal serrations for efficient grasping and coagulation of tissue. It is made of the finest medical grade stainless stell for durability. Surgical grade material is used in its production to provide the highest level of craftsmanship. Each complete set of bipolar forceps includes a handle (this refers to the double action scissor-like curved jaws that grasp), an insulated shaft (a dark or grey plastic-like cylindrical shaft) and inner insert (a complex robotic joint for connecting the jaws/handle to the shaft).',
    'Prograsp Forceps': 'Prograps Forceps are designed meticulously to hold, Grasp and manipulate delicate abdominal tissue during laparoscopic procedures with minimal tissue injury. One of their main functions is to facilitate adequate access to the surgical site by holding tissue to clear off the operating field. This grasper is also known as bowel grasper, bowel tissue is considered to be among the most delicate in the human body, so it is essential that the grasping tip is able to offer a safe, secure grip, without exerting excessive pressure. Its handle is made of double-action straight Jaws with horizontal serration to grasp, hold and manipulate the tissue. Each of the surface of the double-action straight jaws has a stadium-like hole. The Prograsp forceps is made up of the finest quality medical-grade stainless steel for durability. Surgical-grade material is used to provide the highest level of craftsmanship. Each complete set of prograsp forceps includes a handle (this refers to the double action straight jaws with stadium-like holes for grasping), an insulated shaft (a dark or grey plastic-like cylindrical shaft) and inner insert (a complex robotic joint for connecting the jaws/handle to the shaft)',
    'Large Needle Driver': 'The needle driver is used to grasp and manipulate needles to enable free-hand suturing of surgical incisions within the body during laparoscopic procedures. Needle driver jaws are made of high quality tungsten carbide metal which gives very good grip over surgical needles. It consists of a double action straight jaw. Finest quality medical grade stainless steel is being used to make needle drivers for better performance and durability. Surgical grade material being used to provide the highest level of craftsmanship. Each complete set of large needle driver includes a handle (this refers to the double action straight jaws made of tungsten carbide for holding needs), an insulated shaft (a dark or grey plastic-like cylindrical shaft) and inner insert (a complex robotic joint for connecting the jaws/handle to the shaft)',
    'Vessel Sealer': 'The davinci vessel sealer are designed meticulously to hold, and grasp delicate abdominal tissue during laparoscopic procedures with minimal tissue injury. One of their main functions is to apply force to grasp tissues such that the flow of blood in these tissues are restricted. Its handle is made of double action straight Jaws no seration and tapers towards its end. Each of the surface of the double action straight jaws a thin and long rectangular groove with little depths. The vessel sealer is made up of finest quality medical grade stainless steel for durability. Surgical grade material being used to provide the highest level of craftsmanship. Each complete set of vessel sealer includes a handle (this refers to the double action straight jaws with rectangular grooves), an insulated shaft (a dark or grey plastic-like cylindrical shaft) and a inner insert (a complex robotic joint for connecting the jaws/handle to the shaft)',
    'Grasping Retractor': 'Grasping retractors are designed meticulously to hold, Grasp and manipulate delicate abdominal tissue during laparoscopic procedures. One of their main functions is to facilitate adequate access to the surgical site by holding tissue to clear off the operating field. Grasping retractors consists of a double action Jaws with horizontal serration to grasp, hold and manipulate the tissue. Each jaw has two holes, a stadium-like hole and a second smaller circular hole. When closed, the jaws form an oval-like shape. Made up of finest quality medical grade stainless steel for durability. Surgical grade material being used to provide the highest level of craftsmanship. Each complete set of grasping retractor includes a handle (refers to the double action oval jaws with two holes, a stadium like hole and a circular hole), an insulated shaft (a dark or grey plastic-like cylindrical shaft) and inner insert (a complex robotic joint for connecting the jaws/handle to the shaft)',
    'Monopolar Curved Scissors': 'The monopolar curved scissors is a double action scissor that has fine curved jaws with sharp cutting edge for smooth and precise cutting of tissue. Made of the finest medical grade stainless steel for durability. Surgical grade material being used to provide the highest level of craftsmanship. Each complete set of monopoloar curved scissors includes a handle (this refers to the scissor used for cutting), an insulated shaft (a dark or grey plastic-like cylindrical shaft) and inner insert (the inner insert is used for connecting the handle to the shaft. For the Monopolar curved scissors, the inner insert is covered by a plastic tubing)',
    'Other': 'Other instruments.',
}

seg_class_easy_name_dict = {
    'Background': 'Body tissues',
    'Instrument': 'Tool',
    'Shaft': 'Tool body',
    'Wrist': 'Tool neck',
    'Claspers': 'Tool head',
    'Bipolar Forceps': 'Bipolar Forceps',
    'Prograsp Forceps': 'Prograsp Forceps',
    'Large Needle Driver': 'Large Needle Driver',
    'Vessel Sealer': 'Vessel Sealer',
    'Grasping Retractor': 'Grasping Retractor',
    'Monopolar Curved Scissors': 'Monopolar Curved Scissors',
    'Other': 'Other instruments',
}

seg_class_shape_description_dict = {
    'Background': '',
    'Instrument': 'All instruments are cylindrical as a whole, with equipment for tooling at the top.',
    'Shaft': 'The instrument shaft is a cylindrical mettalic instrument.',
    'Wrist': '',
    'Claspers': 'The instrument clasper has a scissor-like appearance.',
    'Bipolar Forceps': '',
    'Prograsp Forceps': '',
    'Large Needle Driver': '',
    'Vessel Sealer': '',
    'Grasping Retractor': '',
    'Monopolar Curved Scissors': '',
    'Other': '',
}

seg_class_easy_description_dict = {
    'Background': '',
    'Instrument': 'tool',
    'Shaft': 'The instrument shaft is a cylindrical mettalic instrument used for carrying a device.',
    'Wrist': 'The instrument wrist is a connector which connects the end-effector (the device at the end of a robotic arm) to the instrument-shaft.',
    'Claspers': 'The primary function of instrument-clasper is the manipulation of tissues. To fulfil this role of manipulation, instrument-clasper has a scissor-like appearance.',
    'Bipolar Forceps': '',
    'Prograsp Forceps': '',
    'Large Needle Driver': '',
    'Vessel Sealer': '',
    'Grasping Retractor': '',
    'Monopolar Curved Scissors': '',
    'Other': '',
}


def get_file_list(in_dir):
    file_list = []
    for root, dirs, files in os.walk(in_dir):
        for filename in files:
            file_list.append(os.path.join(root, filename))
    return file_list
