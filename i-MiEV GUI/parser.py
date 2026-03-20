class Parser:

    @staticmethod
    def parse(data: bytes, state):
        for i in range(len(data) - 4):
            if data[i] == 0xAA:
                can_id = data[i+1] | (data[i+2] << 8)
                value = data[i+3]

                if can_id == 0x0003:
                    state.speed = value
                elif can_id == 0x0001:
                    state.angle = value
                elif can_id == 0x0017:
                    state.brake = value
                elif can_id == 0x0018:
                    state.accel = value
                elif can_id == 0x0004:
                    state.gear = value
