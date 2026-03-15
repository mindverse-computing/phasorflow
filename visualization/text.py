# (c) 2026 Mindverse Computing LLC.
# Licensed under CC BY-NC 4.0.
# See LICENSE file for patent and commercial restrictions.

# phasorflow/visualization/text.py

class TextDrawer:
    @staticmethod
    def draw(circuit):
        """Generates a Qiskit-style ASCII representation of the PhasorCircuit."""
        num_threads = circuit.num_threads
        num_rows = num_threads * 2 - 1
        
        # Initialize rows
        layers = []
        for i in range(num_rows):
            if i % 2 == 0:
                thread_idx = i // 2
                layers.append(f"T{thread_idx}: ──")
            else:
                layers.append("      ") # Padding for 'T0: ──'

        for gate_name, targets, params in circuit.data:
            if gate_name == 'shift':
                block_str = f"[S({params['phi']:.2f})]"
                block_len = len(block_str)
                t = targets[0]
                
                for i in range(num_rows):
                    if i == t * 2:
                        layers[i] += f"{block_str}──"
                    else:
                        char = '─' if i % 2 == 0 else ' '
                        layers[i] += (char * block_len) + (char * 2)
                        
            elif gate_name in ['mix', 'dft']:
                gate_display_name = gate_name.upper()
                block_str = f"[{gate_display_name}]"
                block_len = len(block_str)
                half_len = block_len // 2
                
                min_t = min(targets)
                max_t = max(targets)
                
                # To place the name exactly centered: 
                # (min_t * 2 + max_t * 2) // 2 happens to be exactly min_t + max_t.
                # HOWEVER, if min_t + max_t is odd, it falls on an empty SPACE connector row!
                # If it's even, it falls right on a thread row!
                center_row = min_t + max_t
                
                for i in range(num_rows):
                    if (min_t * 2) <= i <= (max_t * 2):
                        if i == center_row:
                            # It's the exact center where the name goes
                            # But wait, if center is a space row (odd), we need to replace the space
                            # If it's an even row, we replace the line
                            if i % 2 == 0:
                                layers[i] += f"{block_str}──"
                            else:
                                layers[i] += f"{block_str}  "
                        
                        elif i == min_t * 2:
                            # Top wire
                            left = "─" * half_len
                            right = "─" * (block_len - half_len - 1)
                            layers[i] += f"{left}┬{right}──"
                            
                        elif i == max_t * 2:
                            # Bottom wire
                            left = "─" * half_len
                            right = "─" * (block_len - half_len - 1)
                            layers[i] += f"{left}┴{right}──"
                            
                        elif i % 2 == 0:
                            # Intermediate wire
                            left = "─" * half_len
                            right = "─" * (block_len - half_len - 1)
                            layers[i] += f"{left}┼{right}──"
                            
                        else:
                            # Vertical empty space
                            left = " " * half_len
                            right = " " * (block_len - half_len - 1)
                            # center alignment
                            layers[i] += f"{left}│{right}  "
                    else:
                        # Outside the gate span
                        char = '─' if i % 2 == 0 else ' '
                        layers[i] += (char * block_len) + (char * 2)

        # Add termination
        for i in range(num_rows):
            if i % 2 == 0:
                layers[i] += "┤"
                
        return "\n".join(layers)

