import math
import numpy as np
import copy


class Vector:
    def __init__(self, *args):
        if len(args) == 0:
            self.data = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        elif len(args) == 3:
            self.data = np.array([args[0], args[1], args[2]], dtype=np.float64)
        elif len(args) == 1 and isinstance(args[0], Vector):
            self.data = args[0].data.copy()
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            self.data = args[0]
        else:
            raise ValueError("Invalid initialization arguments for Vector")

    def __call__(self, index):
        if 0 <= index <= 2:
            return self.data[index]
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if 0 <= index <= 2:
            self.data[index] = value
        else:
            raise IndexError("Index out of range")

    def __getitem__(self, index):
        return self.__call__(index)

    def __repr__(self):
        return f"{self.data}"

    def x(self, value=None):
        if value != None:
            self.data[0] = value
        else:
            return self.data[0]

    def y(self, value=None):
        if value != None:
            self.data[1] = value
        else:
            return self.data[1]

    def z(self, value=None):
        if value != None:
            self.data[2] = value
        else:
            return self.data[2]

    def reverse_sign(self):
        self.data = [-x for x in self.data]

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self.data = [self.data[i] + other.data[i] for i in range(3)]
        elif isinstance(other, (int, float)):
            self.data = [self.data[i] + other for i in range(3)]
        elif isinstance(other, np.ndarray) and other.size == 3:
            self.data = [self.data[i] + other[i] for i in range(3)]
        else:
            raise ValueError("Invalid addition operand")
        return self

    def __isub__(self, other):
        if isinstance(other, Vector):
            self.data = [self.data[i] - other.data[i] for i in range(3)]
        elif isinstance(other, (int, float)):
            self.data = [self.data[i] - other for i in range(3)]
        elif isinstance(other, np.ndarray) and other.size == 3:
            self.data = [self.data[i] - other[i] for i in range(3)]
        else:
            raise ValueError("Invalid subtraction operand")
        return self

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.data[0] + other.data[0], self.data[1] + other.data[1], self.data[2] + other.data[2])
        elif isinstance(other, (int, float)):
            return Vector(self.data[0] + other, self.data[1] + other, self.data[2] + other)
        elif isinstance(other, np.ndarray) and other.size == 3:
            return Vector(self.data[0] + other[0], self.data[1] + other[1], self.data[2] + other[2])
        else:
            raise ValueError("Invalid addition operand")

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.data[0] - other.data[0], self.data[1] - other.data[1], self.data[2] - other.data[2])
        elif isinstance(other, (int, float)):
            return Vector(self.data[0] - other, self.data[1] - other, self.data[2] - other)
        elif isinstance(other, np.ndarray) and other.size == 3:
            return Vector(self.data[0] - other[0], self.data[1] - other[1], self.data[2] - other[2])
        else:
            raise ValueError("Invalid subtraction operand")

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                self.data[1] * other.data[2] - self.data[2] * other.data[1],
                self.data[2] * other.data[0] - self.data[0] * other.data[2],
                self.data[0] * other.data[1] - self.data[1] * other.data[0],
            )
        elif isinstance(other, (int, float)):
            return Vector(self.data[0] * other, self.data[1] * other, self.data[2] * other)
        else:
            raise ValueError("Invalid multiplication operand")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Vector(self.data[0] / other, self.data[1] / other, self.data[2] / other)

    def __neg__(self):
        return Vector(-self.data[0], -self.data[1], -self.data[2])

    def norm(self, eps=1e-10):
        tmp1 = abs(self.data[0])
        tmp2 = abs(self.data[1])

        if tmp1 >= tmp2:
            tmp2 = abs(self.data[2])
            if tmp1 >= tmp2:
                if tmp1 < eps:
                    return 0
                return tmp1 * math.sqrt(1 + (self.data[1] / self.data[0]) ** 2 + (self.data[2] / self.data[0]) ** 2)
            else:
                return tmp2 * math.sqrt(1 + (self.data[0] / self.data[2]) ** 2 + (self.data[1] / self.data[2]) ** 2)
        else:
            tmp1 = abs(self.data[2])
            if tmp2 > tmp1:
                return tmp2 * math.sqrt(1 + (self.data[0] / self.data[1]) ** 2 + (self.data[2] / self.data[1]) ** 2)
            else:
                return tmp1 * math.sqrt(1 + (self.data[0] / self.data[2]) ** 2 + (self.data[1] / self.data[2]) ** 2)

    def normalize(self, eps=1e-10):
        v = self.norm(eps)
        if v < eps:
            self.data = [1, 0, 0]
            return 0
        else:
            self.data = [x / v for x in self.data]
            return v

    @staticmethod
    def zero():
        return Vector(0, 0, 0)

    def set_2d_xy(self, v):
        self.data[0] = v[0]
        self.data[1] = v[1]
        self.data[2] = 0

    def set_2d_yz(self, v):
        self.data[1] = v[0]
        self.data[2] = v[1]
        self.data[0] = 0

    def set_2d_zx(self, v):
        self.data[2] = v[0]
        self.data[0] = v[1]
        self.data[1] = 0

    @staticmethod
    def equal(a, b, eps=1e-10):
        return all(abs(a.data[i] - b.data[i]) < eps for i in range(3))

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_numpy(self):
        return np.array(self.data)

    def copy(self):
        return copy.deepcopy(self)


class Rotation:
    def __init__(self, *args):
        if len(args) == 0:
            self.data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        elif len(args) == 1:
            self.data = np.array(args[0])

        elif len(args) == 3:
            self.data = np.array(
                [
                    [args[0][0], args[1][1], args[2][2]],
                    [args[1][0], args[1][1], args[1][2]],
                    [args[2][0], args[2][1], args[2][2]],
                ],
                dtype=np.float64,
            )

        elif len(args) == 9:
            self.data = np.array(
                [[args[0], args[1], args[2]], [args[3], args[4], args[5]], [args[6], args[7], args[8]]],
                dtype=np.float64,
            )

    @classmethod
    def identity(cls):
        return cls(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    @classmethod
    def rotX(cls, angle):
        ct = math.cos(angle)
        st = math.sin(angle)
        return cls(1.0, 0.0, 0.0, 0.0, ct, -st, 0.0, st, ct)

    @classmethod
    def rotY(cls, angle):
        ct = math.cos(angle)
        st = math.sin(angle)
        return cls(ct, 0.0, st, 0.0, 1.0, 0.0, -st, 0.0, ct)

    @classmethod
    def rotZ(cls, angle):
        ct = math.cos(angle)
        st = math.sin(angle)
        return cls(ct, -st, 0.0, st, ct, 0.0, 0.0, 0.0, 1.0)

    @classmethod
    def rot(cls, rotvec, angle):
        rotvec = np.array(rotvec, dtype=float)
        rotvec /= np.linalg.norm(rotvec)
        return cls.rot2(rotvec, angle)

    @classmethod
    def rot2(cls, rotvec, angle):
        ct = math.cos(angle)
        st = math.sin(angle)
        vt = 1 - ct
        x, y, z = rotvec

        return cls(
            [
                [ct + vt * x * x, vt * x * y - st * z, vt * x * z + st * y],
                [vt * y * x + st * z, ct + vt * y * y, vt * y * z - st * x],
                [vt * z * x - st * y, vt * z * y + st * x, ct + vt * z * z],
            ]
        )

    @classmethod
    def eulerZYZ(cls, Alfa, Beta, Gamma):
        ca = math.cos(Alfa)
        sa = math.sin(Alfa)
        cb = math.cos(Beta)
        sb = math.sin(Beta)
        cg = math.cos(Gamma)
        sg = math.sin(Gamma)

        return cls(
            [
                [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
                [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
                [-sb * cg, sb * sg, cb],
            ]
        )

    @classmethod
    def rpy(cls, roll, pitch, yaw):
        ca1 = math.cos(yaw)
        sa1 = math.sin(yaw)
        cb1 = math.cos(pitch)
        sb1 = math.sin(pitch)
        cc1 = math.cos(roll)
        sc1 = math.sin(roll)

        Xx = ca1 * cb1
        Yx = ca1 * sb1 * sc1 - sa1 * cc1
        Zx = ca1 * sb1 * cc1 + sa1 * sc1

        Xy = sa1 * cb1
        Yy = sa1 * sb1 * sc1 + ca1 * cc1
        Zy = sa1 * sb1 * cc1 - ca1 * sc1

        Xz = -sb1
        Yz = cb1 * sc1
        Zz = cb1 * cc1

        return cls(Xx, Yx, Zx, Xy, Yy, Zy, Xz, Yz, Zz)

    @classmethod
    def quaternion(cls, x, y, z, w):
        x2, y2, z2, w2 = x * x, y * y, z * z, w * w
        return Rotation(
            [
                [w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2],
            ]
        )

    def __mul__(self, other):
        if isinstance(other, Rotation):
            return Rotation(self.data @ other.data)
        elif isinstance(other, np.ndarray) and other.shape == (3,):
            return self.data @ other
        elif isinstance(other, Vector):
            return Vector(self.data @ other.data)
        else:
            raise TypeError("Multiplication not supported with given type.")

    def transpose(self):
        return self.inverse()

    def inverse(self):
        tmp = Rotation(copy.deepcopy(self.data))
        tmp.set_inverse()
        return tmp

    def set_inverse(self):
        tmp = self.data[0][1]
        self.data[0][1] = self.data[1][0]
        self.data[1][0] = tmp

        tmp = self.data[0][2]
        self.data[0][2] = self.data[2][0]
        self.data[2][0] = tmp

        tmp = self.data[1][2]
        self.data[1][2] = self.data[2][1]
        self.data[2][1] = tmp

    def get_quaternion(self):
        trace = np.trace(self.data)
        epsilon = 1e-12

        if trace > epsilon:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (self.data[2, 1] - self.data[1, 2]) * s
            y = (self.data[0, 2] - self.data[2, 0]) * s
            z = (self.data[1, 0] - self.data[0, 1]) * s
        else:
            if self.data[0, 0] > self.data[1, 1] and self.data[0, 0] > self.data[2, 2]:
                s = 2.0 * math.sqrt(1.0 + self.data[0, 0] - self.data[1, 1] - self.data[2, 2])
                w = (self.data[2, 1] - self.data[1, 2]) / s
                x = 0.25 * s
                y = (self.data[0, 1] + self.data[1, 0]) / s
                z = (self.data[0, 2] + self.data[2, 0]) / s
            elif self.data[1, 1] > self.data[2, 2]:
                s = 2.0 * math.sqrt(1.0 + self.data[1, 1] - self.data[0, 0] - self.data[2, 2])
                w = (self.data[0, 2] - self.data[2, 0]) / s
                x = (self.data[0, 1] + self.data[1, 0]) / s
                y = 0.25 * s
                z = (self.data[1, 2] + self.data[2, 1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + self.data[2, 2] - self.data[0, 0] - self.data[1, 1])
                w = (self.data[1, 0] - self.data[0, 1]) / s
                x = (self.data[0, 2] + self.data[2, 0]) / s
                y = (self.data[1, 2] + self.data[2, 1]) / s
                z = 0.25 * s

        return x, y, z, w

    def do_rotX(self, angle):
        cs = math.cos(angle)
        sn = math.sin(angle)
        x1 = cs * self.data[0, 1] + sn * self.data[0, 2]
        x2 = cs * self.data[1, 1] + sn * self.data[1, 2]
        x3 = cs * self.data[2, 1] + sn * self.data[2, 2]
        self.data[0, 2] = -sn * self.data[0, 1] + cs * self.data[0, 2]
        self.data[1, 2] = -sn * self.data[1, 1] + cs * self.data[1, 2]
        self.data[2, 2] = -sn * self.data[2, 1] + cs * self.data[2, 2]
        self.data[0, 1] = x1
        self.data[1, 1] = x2
        self.data[2, 1] = x3

    def do_rotY(self, angle):
        cs = math.cos(angle)
        sn = math.sin(angle)
        x1 = cs * self.data[0, 0] - sn * self.data[0, 2]
        x2 = cs * self.data[1, 0] - sn * self.data[1, 2]
        x3 = cs * self.data[2, 0] - sn * self.data[2, 2]
        self.data[0, 2] = sn * self.data[0, 0] + cs * self.data[0, 2]
        self.data[1, 2] = sn * self.data[1, 0] + cs * self.data[1, 2]
        self.data[2, 2] = sn * self.data[2, 0] + cs * self.data[2, 2]
        self.data[0, 0] = x1
        self.data[1, 0] = x2
        self.data[2, 0] = x3

    def do_rotZ(self, angle):
        cs = math.cos(angle)
        sn = math.sin(angle)
        x1 = cs * self.data[0, 0] + sn * self.data[0, 1]
        x2 = cs * self.data[1, 0] + sn * self.data[1, 1]
        x3 = cs * self.data[2, 0] + sn * self.data[2, 1]
        self.data[0, 1] = -sn * self.data[0, 0] + cs * self.data[0, 1]
        self.data[1, 1] = -sn * self.data[1, 0] + cs * self.data[1, 1]
        self.data[2, 1] = -sn * self.data[2, 0] + cs * self.data[2, 1]
        self.data[0, 0] = x1
        self.data[1, 0] = x2
        self.data[2, 0] = x3

    def __call__(self, i, j):
        return self.data[i, j]

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"{self.data}"

    def __eq__(self, other):
        if isinstance(other, Rotation):
            return np.allclose(self.data, other.data)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def unitX(self):
        return self.data[:, 0]

    def unitY(self):
        return self.data[:, 1]

    def unitZ(self):
        return self.data[:, 2]

    def get_rpy(self):
        epsilon = 1e-12
        pitch = math.atan2(-self.data[2, 0], math.sqrt(self.data[0, 0] ** 2 + self.data[1, 0] ** 2))

        if abs(pitch) > (math.pi / 2 - epsilon):
            yaw = math.atan2(-self.data[1, 2], self.data[1, 1])
            roll = 0.0
        else:
            roll = math.atan2(self.data[2, 1], self.data[2, 2])
            yaw = math.atan2(self.data[1, 0], self.data[0, 0])

        return roll, pitch, yaw

    def get_eulerZYZ(self):
        epsilon = 1e-12
        if abs(self.data[2, 2]) > 1 - epsilon:
            gamma = 0.0
            if self.data[2, 2] > 0:
                beta = 0.0
                alpha = math.atan2(self.data[1, 0], self.data[0, 0])
            else:
                beta = math.pi
                alpha = math.atan2(-self.data[1, 0], -self.data[0, 0])
        else:
            alpha = math.atan2(self.data[1, 2], self.data[0, 2])
            beta = math.atan2(math.sqrt(self.data[2, 0] ** 2 + self.data[2, 1] ** 2), self.data[2, 2])
            gamma = math.atan2(self.data[2, 0], -self.data[2, 1])

        return alpha, beta, gamma

    def get_eulerZYX(self):
        r, p, y = self.GetRPY()
        return y, p, r

    def get_rot(self):
        axis, angle = self.get_rot_angle()
        return axis * angle

    def get_rot_angle(self, eps=1e-6):
        epsilon = eps
        epsilon2 = eps * 10
        axis = Vector()
        PI = math.pi

        if (
            abs(self.data[0][1] - self.data[1][0]) < epsilon
            and abs(self.data[0][2] - self.data[2][0]) < epsilon
            and abs(self.data[1][2] - self.data[2][1]) < epsilon
        ):
            if (
                abs(self.data[0][1] + self.data[1][0]) < epsilon2
                and abs(self.data[0][2] + self.data[2][0]) < epsilon2
                and abs(self.data[1][2] + self.data[2][1]) < epsilon2
                and abs(self.data[0][0] + self.data[1][1] + self.data[2][2] - 3) < epsilon2
            ):
                axis = Vector(0, 0, 1)
                angle = 0.0
                return axis, angle

            angle = PI
            xx = (self.data[0][0] + 1) / 2
            yy = (self.data[1][1] + 1) / 2
            zz = (self.data[2][2] + 1) / 2
            xy = (self.data[0][1] + self.data[1][0]) / 4
            xz = (self.data[0][2] + self.data[2][0]) / 4
            yz = (self.data[1][2] + self.data[2][1]) / 4

            if xx > yy and xx > zz:
                x = math.sqrt(xx)
                y = xy / x
                z = xz / x
            elif yy > zz:
                y = math.sqrt(yy)
                x = xy / y
                z = yz / y
            else:
                z = math.sqrt(zz)
                x = xz / z
                y = yz / z

            axis = Vector(x, y, z)
            return axis, angle

        f = (self.data[0][0] + self.data[1][1] + self.data[2][2] - 1) / 2

        x = self.data[2][1] - self.data[1][2]
        y = self.data[0][2] - self.data[2][0]
        z = self.data[1][0] - self.data[0][1]
        axis = Vector(x, y, z)
        angle = math.atan2(axis.norm() / 2, f)
        axis.normalize()
        return axis, angle

    def to_numpy(self):
        return np.array(self.data)

    def copy(self):
        return copy.deepcopy(self)


class Frame:
    def __init__(self, *args):
        if len(args) == 2:
            if isinstance(args[0], Rotation) and isinstance(args[1], Vector):
                self.M = args[0]
                self.p = args[1]
            else:
                raise ValueError("Expected (Rotation, Vector) for initialization.")
        elif len(args) == 1:
            if isinstance(args[0], Vector):
                self.p = args[0]
                self.M = Rotation.identity()
            elif isinstance(args[0], Rotation):
                self.M = args[0]
                self.p = Vector.zero()
            else:
                raise ValueError("Expected Vector or Rotation for single argument initialization.")
        else:
            self.p = Vector.zero()
            self.M = Rotation.identity()

    def __repr__(self):
        return f"Frame(Position: {self.p}, Rotation: \n{self.M})"

    def __eq__(self, other):
        if isinstance(other, Frame):
            return self.p == other.p and self.M == other.M
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, i, j):
        if not (0 <= i <= 3 and 0 <= j <= 3):
            raise IndexError("Index out of bounds.")
        if i == 3:
            if j == 3:
                return 1.0
            else:
                return 0.0
        if j == 3:
            return self.p[i]
        return self.M[i, j]

    def inverse(self):
        inv_rot = self.M.inverse()
        inv_pos = -(inv_rot * self.p)
        return Frame(inv_rot, inv_pos)

    def inverse_vector(self, vec):
        return self.M.inverse() * (vec - self.p)

    def __mul__(self, other):
        if isinstance(other, Frame):
            new_p = self.p + self.M * other.p
            new_M = self.M * other.M
            return Frame(new_M, new_p)
        elif isinstance(other, Vector):
            return self.p + self.M * other
        else:
            raise TypeError("Multiplication not supported with given type.")

    @staticmethod
    def identity():
        return Frame(Rotation.identity(), Vector.zero())

    def integrate(self, twist, frequency):
        raise NotImplementedError("Integration not implemented.")

    @staticmethod
    def DH_Craig1989(a, alpha, d, theta):
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        R = Rotation([[ct, -st * ca, st * sa], [st, ct * ca, -ct * sa], [0, sa, ca]])
        p = Vector(a * ct, a * st, d)
        return Frame(R, p)

    @staticmethod
    def DH(a, alpha, d, theta):
        return Frame.DH_Craig1989(a, alpha, d, theta)

    @staticmethod
    def equal(a, b, eps):
        return a.p.equal(b.p, eps) and a.M.equal(b.M, eps)

    def __eq__(self, other):
        if other is None:
            return False
        return self.p == other.p and self.M == other.M

    def __ne__(self, other):
        return not (self == other)

    def to_numpy(self):
        T = np.eye(4)
        T[:3, :3] = self.M.to_numpy()
        T[:3, 3] = self.p.to_numpy()
        return T

    def copy(self):
        return copy.deepcopy(self)
