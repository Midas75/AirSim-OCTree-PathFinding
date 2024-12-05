from __future__ import annotations
import socket
import numpy
import cv2
import json
import typing
import struct


class Annotation:
    xyxy: list[float]
    name: str
    conf: float
    visible: bool
    distance: float

    def __init__(
        self,
        annotationDict: dict[str, typing.Any] = {},
        xyxy: list[float] = [],
        name: str = "",
        visible: bool = True,
        distance: float = 0,
        conf: float = 1,
    ) -> None:
        if len(annotationDict) != 0:
            self.xyxy = annotationDict["xyxy"]
            self.name = annotationDict["name"]
            self.visible = annotationDict["visible"]
            self.distance = annotationDict["distance"]
        else:
            self.xyxy = xyxy
            self.visible = visible
            self.name = name
            self.distance = distance
            self.conf = conf


class DebugData:
    forward: tuple[float, float, float]

    def __init__(self, data: bytes) -> None:
        self.forward = struct.unpack(">3f", data[0:12])


class Attitude:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    rw: float
    fov: float

    def __init__(
        self,
        data: typing.Union[
            tuple[float, float, float, float, float, float, float, float], bytes
        ],
    ) -> None:
        if isinstance(data, ((bytes, bytearray), bytearray)):
            self.x, self.y, self.z, self.rx, self.ry, self.rz, self.rw, self.fov = (
                struct.unpack(">8f", data)
            )
        else:
            self.x = data[0]
            self.y = data[1]
            self.z = data[2]
            self.rx = data[3]
            self.ry = data[4]
            self.rz = data[5]
            self.rw = data[6]
            self.fov = data[7]

    def getPosition(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        return f"{self.x},{self.y},{self.z},{self.rx},{self.ry},{self.rz},{self.rw},{self.fov}"


class Depth:
    col: int
    row: int
    depths: list[list[float]]

    def __init__(
        self, data: typing.Union[bytes, tuple[int, int, list[list[float]]]]
    ) -> None:
        if isinstance(data, (bytes, bytearray)):
            self.row = int.from_bytes(data[0:4], "big")
            self.col = int.from_bytes(data[4:8], "big")
            self.depths = list[list[float]]()
            for i in range(self.row):
                self.depths.append(
                    list(
                        struct.unpack(
                            f">{self.col}f",
                            data[8 + i * self.col * 4 : 8 + (i + 1) * self.col * 4],
                        )
                    )
                )
        else:
            self.row = data[0]
            self.col = data[1]
            self.depths = data[2]

    def show(self, ratio: int = 60) -> None:
        img = numpy.ones((self.col * ratio, self.row * ratio, 3), numpy.uint8) * 255
        for i in range(self.col):
            for j in range(self.row):
                cv2.putText(
                    img,
                    f"{self.depths[j][i]:.2f}",
                    (i * ratio + ratio // 2, self.col * ratio - j * ratio - ratio // 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (128, 0, 0),
                    1,
                )
        cv2.imshow("depth", img)
        cv2.waitKey(1)


class Image:
    rawBytes: typing.Union[bytearray, bytes]
    image: numpy.ndarray
    annotations: list[Annotation]

    def __init__(
        self, data: typing.Union[bytes, Image], postProcess: bool = False
    ) -> None:
        if isinstance(data, Image):
            self.rawBytes = data.rawBytes.copy()
            self.image = data.image.copy()
            self.annotations = data.annotations.copy()
        elif isinstance(data, (bytes, bytearray)):
            self.rawBytes = data
            n = numpy.frombuffer(data, dtype=numpy.uint8)
            self.image = cv2.imdecode(n, cv2.IMREAD_COLOR)
            if postProcess:
                self.image = self.image[::-1, :, :]
                self.image = self.image.copy()
                self.image = self._gammaCorrect(self.image)
            else:
                self.image = self.image.copy()
            self.annotations = []
            rects, l = self._reverseParseJson(data)
            if l != -1:
                rects = json.loads(rects)
                for item in rects:
                    self.annotations.append(Annotation(item))
        else:
            raise TypeError()

    def _reverseParseJson(self, data: bytes) -> tuple[bytes, int]:
        pos = data.find(b"\xFF\xD9")
        if pos == len(data) - 2:
            return [], -1
        return data[pos + 2 :], pos

    def _gammaCorrect(self, image: numpy.ndarray, gamma: float = 2.2) -> numpy.ndarray:
        return (((image / 255) ** (1 / gamma)) * 255).astype("uint8")

    def show(
        self, withAnnotation: bool = True, depths: Depth = None, title: str = "image"
    ):
        show_image = self.image.copy()
        if withAnnotation:
            for ann in self.annotations:
                if ann.visible:
                    cv2.rectangle(
                        show_image,
                        (
                            int(ann.xyxy[0] * show_image.shape[1]),
                            int(ann.xyxy[1] * show_image.shape[0]),
                        ),
                        (
                            int(ann.xyxy[2] * show_image.shape[1]),
                            int(ann.xyxy[3] * show_image.shape[0]),
                        ),
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        show_image,
                        f"{ann.name} {ann.distance:.2f}",
                        (
                            int(ann.xyxy[0] * show_image.shape[1]),
                            int(ann.xyxy[1] * show_image.shape[0] + 15),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
        if depths != None:
            block = (
                numpy.ones(
                    (
                        int(show_image.shape[0] / depths.row),
                        int(show_image.shape[1] / depths.col),
                        3,
                    ),
                    numpy.uint8,
                )
                * 255
            )
            for i in range(depths.col):  # col
                for j in range(depths.row):  # row
                    l = int(i / depths.col * show_image.shape[1]) + 1
                    r = int((i + 1) / depths.col * show_image.shape[1]) - 1
                    t = int(j / depths.row * show_image.shape[0]) + 1
                    b = int((j + 1) / depths.row * show_image.shape[0]) - 1
                    alpha = depths.depths[depths.row - j - 1][i] / 100
                    new_block = cv2.addWeighted(
                        show_image[t:b, l:r],
                        1 - alpha,
                        block[: b - t, : r - l],
                        alpha,
                        0,
                    )
                    show_image[t:b, l:r] = new_block
        cv2.imshow(title, show_image)
        cv2.waitKey(1)


class DroneRequestClient:
    _client: socket.socket
    _cacheId: int = -1
    _cacheIdBytes: bytes

    def __init__(self, address: tuple[str, int] = ("127.0.0.1", 8811)) -> None:
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.connect(address)

    def _prependLength(self, data: bytes) -> bytes:
        length = len(data).to_bytes(4, byteorder="big")
        return length + data

    def _recvWithLength(self, length: int) -> bytearray:
        buffer = bytearray()
        while True:
            buffer += self._client.recv(length - len(buffer))
            if len(buffer) >= length:
                break
        return buffer

    def _recvIntLength(self) -> int:
        return int.from_bytes(self._client.recv(4), "big")

    def _recvBytes(self) -> bytearray:
        return self._recvWithLength(self._recvIntLength())

    def _getCacheIdBytes(self, id: int) -> bytes:
        if id == self._cacheId:
            pass
        else:
            self._cacheId = id
            self._cacheIdBytes = self._cacheId.to_bytes(1, "big")
        return self._cacheIdBytes

    def getImage(self, id: int, withAnnotations: bool = False) -> Image:
        requestData = self._prependLength(
            b"\x01"
            + self._getCacheIdBytes(id)
            + (b"\x01" if withAnnotations else b"\x00")
        )
        self._client.send(requestData)
        return Image(self._recvBytes())

    def getDepth(self, id: int) -> Depth:
        requestData = self._prependLength(b"\x05" + self._getCacheIdBytes(id))
        self._client.send(requestData)
        return Depth(self._recvBytes())

    def getAttitude(self, id: int) -> Attitude:
        requestData = self._prependLength(b"\x06" + self._getCacheIdBytes(id))
        self._client.send(requestData)
        return Attitude(self._recvBytes())

    def getDebugData(self, id: int) -> DebugData:
        requestdata = self._prependLength(b"\x07" + self._getCacheIdBytes(id))
        self._client.send(requestdata)
        return DebugData(self._recvBytes())

    def __del__(self):
        self._client.close()


if __name__ == "__main__":
    rc = DroneRequestClient(("127.0.0.1", 8811))

    while True:
        rc.getImage(1).show()
