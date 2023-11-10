from matplotlib.animation import FuncAnimation

import amc.amc_parser as amc_parser
from motiflets.plotting import *

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150


def get_joint_pos_dict(c_joints, c_motion):
    c_joints['root'].set_motion(c_motion)
    out_dict = {}
    for k1, v1 in c_joints['root'].to_dict().items():
        for k2, v2 in zip('xyz', v1.coordinate[:, 0]):
            out_dict['{}_{}'.format(k1, k2)] = v2
    return out_dict


def exclude_body_joints(df):
    # Filter body joints as suggested by Yeh
    exclude = ['root', 'lowerback', 'upperback',
               'thorax', 'lowerneck', 'upperneck', 'head']
    exclude_bones = []
    exclude_bones.extend([x + "_" + k for x in exclude for k in 'xyz'])
    exclude_bones

    return df[~df.index.isin(exclude_bones)]


def include_joints(df, include, add_xyz=True):
    include_bones = []

    if add_xyz:
        include_bones.extend([x + "_" + k for x in include for k in 'xyz'])
    else:
        include_bones = include

    return df[df.index.isin(include_bones)]


def draw_frame(ax, motions, joints, i, joints_to_highlight=None):
    ax.cla()
    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    joints['root'].set_motion(motions[i])

    c_joints = joints['root'].to_dict()
    # xs, ys, zs, color = [], [], [], []
    for joint in c_joints.values():
        xs = (joint.coordinate[0, 0])
        ys = (joint.coordinate[1, 0])
        zs = (joint.coordinate[2, 0])
        color = 'r.' if joint.name in joints_to_highlight else 'b.'

        ax.plot(zs, xs, ys, color)

    for joint in c_joints.values():
        child = joint
        if child.parent is not None:
            parent = child.parent
            xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
            ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
            zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]

            color = 'r' if (child.name in joints_to_highlight) and (
                    parent.name in joints_to_highlight) else 'b'
            ax.plot(zs, xs, ys, color)


# plot multi-variat motiflet
def plot_multivariate_motiflet(
        data, motifset, m, d=[], names=[]
):
    fig, axes = plt.subplots(len(data) + 1, 1, figsize=(14, 2 * len(data)))

    for i in range(len(data)):
        ax = axes[i]
        # plt.subplot(len(data) + 1, 1, i + 1)
        if len(names) == len(data):
            ax.set_title('${{{0}}}$'.format(names[i]))
        else:
            ax.set_title('$T_{{{0}}}$'.format(i + 1))

        for idx, pos in enumerate(motifset):
            ax.plot(range(0, m), data[i, :][pos:pos + m])  # c=color[idx])

        ax.set_xlim((0, m))

    plt.tight_layout()
    plt.show()


# http://mocap.cs.cmu.edu/search.php?subjectnumber=13&motion=%

datasets = {
    "Boxing": {
        "ks": 15,
        "motif_length": 100,
        "amc_name": "13_17",
        "asf_path": '../datasets/motion_data/13.asf'
    },
    "Charleston-Fancy": {
        "ks": 15,
        "motif_length": 120,
        "amc_name": "93_08",  # Fancy Charleston
        "asf_path": '../datasets/motion_data/93.asf'
    },
    "Charleston-Side-By-Side-Female": {
        "ks": 15,
        "motif_length": 120,
        "amc_name": "93_04",
        "asf_path": '../datasets/motion_data/93.asf'
    },
    "Charleston-Side-By-Side-Male": {
        "ks": 15,
        "motif_length": 120,
        "amc_name": "93_05",
        "asf_path": '../datasets/motion_data/93.asf'
    }
}

dataset = datasets["Boxing"]
k_max = dataset["ks"]
motif_length = dataset["motif_length"]
amc_name = dataset["amc_name"]
asf_path = dataset["asf_path"]
amc_path = '../datasets/motion_data/' + amc_name + '.amc'

# use_joints = np.asarray(
#    ['root', 'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 'head',
#     'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb',
#     'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb',
#     'rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes'])

# Body
# use_joints = ['rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes']

# Right
# use_joints = ['rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb']

# use_joints = [  'lhand', 'lfingers', 'lthumb'
#               'rhand', 'rfingers', 'rthumb']

use_joints = ['rclavicle', 'rhumerus', 'rradius', 'rwrist',
             'rhand', 'rfingers', 'rthumb',
             'rfemur', 'rtibia', 'rfoot', 'rtoes']


# footwork
# use_joints = ['rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes']


def test_plotting():
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = exclude_body_joints(df)
    # df = include_joints(df, use_joints)
    # print("Used joints:", use_joints)

    print("Data", df.shape)

    series = df

    length_range = np.arange(50, 200, 10)
    print(length_range)

    ml = Motiflets(amc_name, series,
                   dimension_labels=df.index,
                   n_dims=10
                   )

    m, all_minima = ml.fit_motif_length(
        k_max, length_range,
        plot=True,
        plot_best_only=False,
        plot_motifsets=True)

    for minimum in all_minima:
        motif_length = length_range[minimum]
        dists = ml.all_dists[minimum]
        elbow_points = ml.all_elbows[minimum]

        # motiflets = ml.all_top_motiflets[minimum]
        motiflets = np.zeros(len(dists), dtype=np.object)
        motiflets[elbow_points] = ml.all_top_motiflets[minimum]

        dimensions = np.zeros(len(dists), dtype=np.object)
        dimensions[elbow_points] = ml.all_dimensions[minimum] # need to unpack

        #dists2, motiflets2, elbow_points2 = ml.fit_k_elbow(
        #    k_max,
        #    plot_elbows=False,
        #    plot_motifs_as_grid=True,
        #    motif_length=motif_length)


        video = True
        if video:
            if len(elbow_points) > 1:
                for eb in elbow_points:
                    for i, pos in enumerate(motiflets[eb]):
                        use_joints = df.index.values[dimensions[eb]]  # FIXME!?
                        # strip the _x, _y, _z from the joint
                        use_joints = [joint[:-2] for joint in use_joints]
                        fig = plt.figure()
                        ax = plt.axes(projection='3d')

                        out_path = ('video/motiflet_' + amc_name + '_' + str(motif_length)
                                    + '_' + str(eb) + '_' + str(i) + '.gif')

                        FuncAnimation(fig,
                                      lambda i: draw_frame(
                                                ax, motions, joints, i,
                                                joints_to_highlight=use_joints
                                            ),
                                      range(pos, pos + motif_length, 4)).save(
                            out_path,
                            bitrate=1000,
                            fps=20)


def test_motion_capture():
    _generate_motion_capture(use_joints)


def _generate_motion_capture(joints_to_use, prefix=None, add_xyz=True):
    joints = amc_parser.parse_asf(asf_path)
    motions = amc_parser.parse_amc(amc_path)

    df = pd.DataFrame(
        [get_joint_pos_dict(joints, c_motion) for c_motion in motions]).T
    df = exclude_body_joints(df)
    df = include_joints(df, joints_to_use, add_xyz=add_xyz)

    print("Used joints:", joints_to_use)
    series = df.values

    ml = Motiflets(amc_name, series,
                   # elbow_deviation=1.25,
                   slack=0.5,
                   dimension_labels=df.index
                   )

    dists, candidates, elbow_points = ml.fit_k_elbow(
        k_max,
        plot_elbows=True,
        motif_length=motif_length)

    print("----")
    print(dists)
    print(elbow_points)
    print(list(candidates[elbow_points]))
    print("----")

    path_ = "video/motiflet_" + amc_name + "_Channels_" + str(
        len(df.index)) + "_Motif.pdf"
    ml.plot_motifset(path=path_)

    motiflets = candidates[elbow_points]
    if add_xyz:
        filtered_joints = joints_to_use
    else:
        filtered_joints = list(set([joint[:-2] for joint in joints_to_use]))

    for i, motiflet in enumerate(motiflets):
        for j, pos in enumerate(motiflet):
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            if prefix:
                out_path = 'video/motiflet_' + amc_name + '_' + prefix + '_' \
                           + str(i) + '_' + str(j) + '.gif'
            else:
                out_path = 'video/motiflet_' + amc_name + '_' \
                           + str(i) + '_' + str(j) + '.gif'

            FuncAnimation(fig,
                          lambda i: draw_frame(ax, motions, joints, i,
                                               joints_to_highlight=filtered_joints),
                          range(pos, pos + motif_length, 4)).save(
                out_path,
                bitrate=1000,
                fps=20)

