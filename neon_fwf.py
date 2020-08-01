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
    f_asc = pd.read_csv(path, skiprows = 25,header = None, nrows = 8000020) # 100024, 5000010
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
        if sgm[i].shape[0] > 2:
            dse = []
            for j in range(len(sgm[i]) - 1):
                d = data[i][sgm[i][1:][j]]
                dse.append(d)
            rrr = np.array(dse)
        else:
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
            t1 = []
            for j in range(len(rtn_all[i])):
                by = np.array(rtn_all[i][j][0].split()).astype('int')
                t1.append(by)
            t = np.array(t1)
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
    lst_rtn = data_extraction_[4][:, 1]
    ouinp, rtinp, _, _, _ = data_extraction(path)
    ou_zr = []
    for i in range(len(ouinp)):
        zro = np.zeros(ouinp[i].shape)
        ouinp_ = np.c_[ouinp[i], zro]
        ou_zr.append(ouinp_)
    ou_zrnp = np.array(ou_zr)
    rto = []
    for i in range(len(rtinp)):
        if len(rtinp[i]) > 5:
            one = np.ones(rtinp[i].shape)
            rtinp_ = np.c_[rtinp[i], one]
        else:
            rrt0 = []
            rrt1 = []
            for j in range(len(rtinp[i])):
                if j == 0:
                    ones = np.ones(rtinp[i][j].shape)
                    tt0 = np.c_[rtinp[i][j], ones]
                    rrt0.append(tt0)
                elif j == 1:
                    two = np.ones(rtinp[i][j].shape) * 2
                    tt1 = np.c_[rtinp[i][j], two]
                    rrt1.append(tt1)
            rrt_cncl = []
            for k in range(len(rrt0)):
                rrt_cnc_ = np.concatenate((rrt0[k], rrt1[k]))
                rrt_cncl.append(rrt_cnc_)
            rtinp_ = np.array(rrt_cncl)[0]
        rto.append(rtinp_)
    rto_np = np.array(rto)
    conc = []
    for i in range(len(ouinp)):
        cnc = np.concatenate((ou_zrnp[i], rto_np[i]))
        conc.append(cnc)
    conc_np = np.array(conc)
    x_w, y_w, z_w = [], [], []
    for i in range(len(conc_np)):
        otg = conc_np[i][conc_np[i][:, 1] == 0]
        rtg = conc_np[i][conc_np[i][:, 1] == 1]
        rtg1 = conc_np[i][conc_np[i][:, 1] == 2]
        ox_ = dx[i] * np.arange(otg.shape[0])
        oy_ = dy[i] * np.arange(otg.shape[0])
        oz_ = dz[i] * np.arange(otg.shape[0])
        x_w_o = x_an[i] + ox_
        y_w_o = y_an[i] + oy_
        z_w_o = z_an[i] + oz_
        frx_ = dx[i] * (first_rtn[i] + np.arange(rtg.shape[0]))
        fry_ = dy[i] * (first_rtn[i] + np.arange(rtg.shape[0]))
        frz_ = dz[i] * (first_rtn[i] + np.arange(rtg.shape[0]))
        frx1_ = dx[i] * (lst_rtn[i] - np.arange(rtg1.shape[0]))
        fry1_ = dy[i] * (lst_rtn[i] - np.arange(rtg1.shape[0]))
        frz1_ = dz[i] * (lst_rtn[i] - np.arange(rtg1.shape[0]))
        x_w_r = x_an[i] + frx_
        y_w_r = y_an[i] + fry_
        z_w_r = z_an[i] + frz_
        x1_w_r = (x_an[i] + frx1_)[::-1]
        y1_w_r = (y_an[i] + fry1_)[::-1]
        z1_w_r = (z_an[i] + frz1_)[::-1]
        x_w_ = np.concatenate((x_w_o, np.concatenate((x_w_r, x1_w_r))))
        y_w_ = np.concatenate((y_w_o, np.concatenate((y_w_r, y1_w_r))))
        z_w_ = np.concatenate((z_w_o, np.concatenate((z_w_r, z1_w_r))))
        x_w.append(x_w_), y_w.append(y_w_), z_w.append(z_w_)
    x_w_np = np.concatenate(np.array(x_w))
    y_w_np = np.concatenate(np.array(y_w))
    z_w_np = np.concatenate(np.array(z_w))
    points = np.transpose((x_w_np, y_w_np, z_w_np))
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
        if len(rtinp[i]) > 5:
            one = np.ones(rtinp[i].shape)
            rtinp_ = np.c_[rtinp[i], one]
        else:
            two = np.ones(np.concatenate(rtinp[i]).shape) * 2
            rtinp_ = np.c_[np.concatenate(rtinp[i]), two]
        rto.append(rtinp_)
    rto_np = np.array(rto)
    conc = []
    for i in range(len(ouinp)):
        cnc = np.concatenate((ou_zrnp[i], rto_np[i]))
        conc.append(cnc)
    conc_np = np.array(conc)
    amplitude = np.vstack((conc_np))
    prv = 0
    idx = [0]
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
