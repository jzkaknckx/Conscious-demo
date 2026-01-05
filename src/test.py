
loop_radius = 2

cx = 216.0
cy = 180.0
visited = [(cx, cy)]
visited.append((232.0, 164.0))
visited.append((216.0, 180.0))

cx, cy = visited[-1]
for (vx, vy) in visited[:-1]:
    dx = cx - vx; dy = cy - vy
    if (dx * dx + dy * dy) <= (loop_radius * loop_radius):
        print(1)