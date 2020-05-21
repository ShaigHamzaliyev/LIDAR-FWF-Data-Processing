def header(file_path):
    import pandas as pd
    header_df = pd.read_csv(file_path, nrows = 25, header = None)
    header_df = header_df[0].str.rsplit(':', expand = True)
    file_signature = header_df[1].iloc[0]
    global_parameters = float(header_df[1].iloc[1])
    file_source_ID = float(header_df[1].iloc[2])
    project_ID_GUID = header_df[1].iloc[3]
    system_identifier = header_df[1].iloc[4]
    generating_software = header_df[1].iloc[5]
    file_creation_day_year = header_df[1].iloc[6]
    version = float(header_df[1].iloc[7])
    header_size = float(header_df[1].iloc[8])
    offset_to_pulse_data = float(header_df[1].iloc[9])
    number_of_pulses = float(header_df[1].iloc[10])
    pulse_format = float(header_df[1].iloc[11])
    pulse_attributes = float(header_df[1].iloc[12])
    pulse_size = float(header_df[1].iloc[13])
    pulse_compression = float(header_df[1].iloc[14])
    reserved = float(header_df[1].iloc[15])
    number_of_vlrs = float(header_df[1].iloc[16])
    number_of_avlrs = float(header_df[1].iloc[17])
    scale_factor_t = float(header_df[1].iloc[18])
    t_offset = float(header_df[1].iloc[19])
    min_T = float(header_df[0].iloc[20].split()[2])
    max_T = float(header_df[0].iloc[20].split()[3])
    x_scale_factor = float(header_df[1].iloc[21].split()[0])
    y_scale_factor = float(header_df[1].iloc[21].split()[1])
    z_scale_factor = float(header_df[1].iloc[21].split()[2])
    x_offset = float(header_df[1].iloc[22].split()[0])
    y_offset = float(header_df[1].iloc[22].split()[1])
    z_offset = float(header_df[1].iloc[22].split()[2])
    min_x = float(header_df[1].iloc[23].split()[0])
    min_y = float(header_df[1].iloc[23].split()[1])
    min_z = float(header_df[1].iloc[23].split()[2])
    max_x = float(header_df[1].iloc[24].split()[0])
    max_y = float(header_df[1].iloc[24].split()[1])
    max_z = float(header_df[1].iloc[24].split()[2])
    return [x_scale_factor, y_scale_factor, z_scale_factor, x_offset, y_offset, z_offset, file_signature, global_parameters, 
            file_source_ID, project_ID_GUID, system_identifier, generating_software, file_creation_day_year, 
            version, header_size, offset_to_pulse_data, number_of_pulses, pulse_format, pulse_attributes, 
           pulse_size, pulse_compression, reserved, number_of_vlrs, number_of_avlrs, scale_factor_t, t_offset, 
            min_T, max_T, min_x, max_x, min_y, max_y, min_z, max_z]



def data_extraction(path):
    import numpy as np
    import pandas as pd
    f_asc = pd.read_csv(path, skiprows = 25,header = None, nrows = 12000)
    data = np.array(np.split(f_asc.to_numpy(), np.where(f_asc.to_numpy()[:, 0] == 'P')[0]))[1:]
    rtn = []
    rtn_all = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if len(data[i][j][0].split()) > 10:
                rtn.append(j)
    rtnp = np.array(rtn)
    sgm = np.array(np.split(rtnp, np.where(rtnp == 12)[0])[1:])
    rtnall = []
    xyz_anc = []
    xyz_trg = []
    frst_lst = []
    outg = []
    for i in range(len(data)):
        an = data[i][3]
        trg = data[i][4]
        frls = data[i][5]
        rrr = data[i][sgm[i][1:]]
        ou = data[i][sgm[i][:1]]
        rtnall.append(rrr)
        xyz_anc.append(an)
        xyz_trg.append(trg)
        frst_lst.append(frls)
        outg.append(ou)
    rtn_all = np.array(rtnall)
    outg = np.array(outg)
    tb = []
    for i in range(len(rtn_all)):
        if len(rtn_all[i]) < 2:
            t = np.array(rtn_all[i][0][0].split()).astype(int)
        else:
            t = np.array(np.sum(rtn_all[i]).split()).astype(int)
        tb.append(t)
    ob = []
    for i in range(len(outg)):
        t = np.array(np.sum(outg[i]).split()).astype(int)
        ob.append(t)
    ouinp = np.array(ob)
    rtinp = np.array(tb)
    xyz_anc_np = np.array(xyz_anc)
    xyz_trg_np = np.array(xyz_trg)
    frst_lst_np = np.array(frst_lst)
    anch = []
    trgt = []
    frls = []
    for i in range(len(xyz_anc_np)):
        ff = np.array(np.squeeze(xyz_anc_np)[i].split()).astype(int)
        dd = np.array(np.squeeze(xyz_trg_np)[i].split()).astype(int)
        ss = np.array(np.squeeze(frst_lst_np)[i].split()).astype(int)
        anch.append(ff)
        trgt.append(dd)
        frls.append(ss)
    anch = np.array(anch)
    trgt = np.array(trgt)
    frls = np.array(frls)
    return ouinp, rtinp, anch, trgt, frls
    
def georeferencing(path):
    import numpy as np
    header_ = header(path)
    data_extraction_ = data_extraction(path)
    x_scl = header_[0]
    y_scl = header_[1]
    z_scl = header_[2]
    x_off = header_[3]
    y_off = header_[4]
    z_off = header_[5]
    x_trg = data_extraction_[3][:, 0] * x_scl + x_off
    y_trg = data_extraction_[3][:, 1] * y_scl + y_off
    z_trg = data_extraction_[3][:, 2] * z_scl + z_off
    x_an = data_extraction_[2][:, 0] * x_scl + x_off
    y_an = data_extraction_[2][:, 1] * y_scl + y_off
    z_an = data_extraction_[2][:, 2] * z_scl + z_off
    dx = (x_trg - x_an) / 1000
    dy = (y_trg - y_an) / 1000
    dz = (z_trg - z_an) / 1000
    first_rtn = data_extraction_[4][:, 0]
    ouinp, rtinp, _, _, _ = data_extraction(path)
    ou_zr = []
    for i in range(len(ouinp)):
        zro = np.zeros(ouinp[i].shape)
        ouinp_ = np.c_[ouinp[i], zro]
        ou_zr.append(ouinp_)
    ou_zrnp = np.array(ou_zr)
    rto = []
    for i in range(len(rtinp)):
        one = np.ones(rtinp[i].shape)
        rtinp_ = np.c_[rtinp[i], one]
        rto.append(rtinp_)
    rto_np = np.array(rto)
    conc = []
    for i in range(len(ouinp)):
        cnc = np.concatenate((ou_zrnp[i], rto_np[i]))
        conc.append(cnc)
    conc_np = np.array(conc)
    first_rtn_ = []
    for i in range(len(data_extraction_[0])):
        dd = first_rtn[i] + np.arange(conc_np[i][:, 0].shape[0])
        first_rtn_.append(dd)
    first_rtn_ar = np.array(first_rtn_)
    x_w = np.concatenate((x_an + first_rtn_ar * dx))
    y_w = np.concatenate((y_an + first_rtn_ar * dy))
    z_w = np.concatenate((z_an + first_rtn_ar * dz))
    points = np.transpose((x_w, y_w, z_w))
    return points
    
    
def create_hdf(path, name_hdf):
    import numpy as np
    import h5py
    ouinp, rtinp, anch, trgt, frls = data_extraction(path)
    ou_zr = []
    for i in range(len(ouinp)):
        zro = np.zeros(ouinp[i].shape)
        ouinp_ = np.c_[ouinp[i], zro]
        ou_zr.append(ouinp_)
    ou_zrnp = np.array(ou_zr)
    rto = []
    for i in range(len(rtinp)):
        one = np.ones(rtinp[i].shape)
        rtinp_ = np.c_[rtinp[i], one]
        rto.append(rtinp_)
    rto_np = np.array(rto)
    conc = []
    for i in range(len(ouinp)):
        cnc = np.concatenate((ou_zrnp[i], rto_np[i]))
        conc.append(cnc)
    conc_np = np.array(conc)
    amplitude = np.vstack((conc_np))
    prv = 0
    idx = []
    for i in range(len(conc_np)):
        shp = conc_np[i].shape[0]
        prv += shp
        idx.append(prv)
    Index = np.array(idx)
    XYZ = georeferencing(path)
    f = h5py.File(name_hdf, 'w')
    f.create_dataset('Amplitude', data = amplitude, dtype='i')
    f.create_dataset('Index', data = Index, dtype='i')
    f.create_dataset('XYZ', data = XYZ) 
