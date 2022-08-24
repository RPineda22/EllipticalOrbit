import unreal
import math
import numpy as np
import pandas as pd


def calculateTstar(rz, dt):
    newT = rz + dt / 2
    return newT


def calculate_rx_star(rx, vx, dt):
    new_rx_star = rx + (vx * dt / 2)
    return new_rx_star


def calculate_ry_star(ry, vy, dt):
    new_ry_star = ry + (vy * dt / 2)
    return new_ry_star


def calculate_rz_star(rz, vz, dt):
    newRZ = rz + (vz * dt / 2)
    return newRZ


def calculate_vx_star(vx, dt, G, M, rx, ry, rz):
    new_vx_star = vx - (dt / 2) * (G * M / (math.sqrt(rx ** 2 + ry ** 2 + rz ** 2) ** 3) * rx)
    return new_vx_star


def calculate_vy_star(vy, dt, G, M, rx, ry, rz):
    new_vy_star = vy - (dt / 2) * (G * M / (math.sqrt(rx ** 2 + ry ** 2 + rz ** 2) ** 3) * ry)
    return new_vy_star


def calculate_vz_star(vz, dt, G, M, rx, ry, rz):
    new_vz_star = vz - (dt / 2) * (G * M / (math.sqrt(rx ** 2 + ry ** 2 + rz ** 2) ** 3) * rz)
    return new_vz_star


def calculate_t(rz, dt):
    new_t = rz + dt
    return new_t


def calculate_rx(rx, vx_star, dt):
    new_rx = rx + vx_star * dt
    return new_rx


def calculate_ry(ry, vy_star, dt):
    new_ry = ry + vy_star * dt
    return new_ry


def calculate_rz(rz, vz_star, dt):
    new_rz = rz + vz_star * dt
    return new_rz


def calculate_vx(vx, G, M, rx_star, ry_star, rz_star, dt):
    new_vx = vx - (G * M / math.sqrt(rx_star ** 2 + ry_star ** 2 + rz_star ** 2) ** 3 * rx_star * dt)
    return new_vx


def calculate_vy(vy, G, M, rx_star, ry_star, rz_star, dt):
    new_vy = vy - (G * M / math.sqrt(rx_star ** 2 + ry_star ** 2 + rz_star ** 2) ** 3 * ry_star * dt)
    return new_vy


def calculate_vz(vz, G, M, rx_star, ry_star, rz_star, dt):
    new_vz = vz - (G * M / (math.sqrt(rx_star ** 2 + ry_star ** 2 + rz_star ** 2) ** 3) * rz_star * dt)
    return new_vz


def calculate_orbit(n_steps):
    columns = 14
    rows = n_steps+1
    my_array = np.ndarray(shape=(rows, columns))

    G = 6.6743 * (10 ** -11)
    M = 5.972 * (10 ** 24)
    dt = 0.05

    my_array[0, 0] = 0.05
    my_array[0, 1] = 0
    my_array[0, 2] = 0
    my_array[0, 3] = 0
    my_array[0, 4] = 0
    my_array[0, 5] = 0
    my_array[0, 6] = 0
    my_array[0, 7] = 0
    my_array[0, 8] = 7000000
    my_array[0, 9] = 0
    my_array[0, 10] = 0
    my_array[0, 11] = 0
    my_array[0, 12] = 5000
    my_array[0, 13] = 0

    # Iterating through equation and storing it to array
    for row in range(1, rows):
        i = row

        for column in range(columns):

            if column == 0:
                # calculating t star
                rz = my_array[i-1, column]
                newT = rz + .10
                my_array[i, column] = newT

            elif column == 1:
                # calculating rx star
                new_rx_star = calculate_rx_star(my_array[row-1, 8], my_array[row-1, 11], dt)
                my_array[row, column] = new_rx_star

            elif column == 2:
                # calculating ry star
                ry = my_array[row-1, 9]
                vy = my_array[row-1, 12]
                new_ry_star = calculate_ry_star(ry, vy, dt)
                my_array[row, column] = new_ry_star

            elif column == 3:
                # calculating rz star
                rz = my_array[row-1, 10]
                vz = my_array[row-1, 13]
                new_rz_star = calculate_rz_star(rz, vz, dt)
                my_array[row, column] = new_rz_star

            elif column == 4:
                # calculating vx star
                vx = my_array[row-1, 11]
                rx = my_array[row-1, 8]
                ry = my_array[row-1, 9]
                rz = my_array[row-1, 10]
                new_vx_star = calculate_vx_star(vx, dt, G, M, rx, ry, rz)
                my_array[row, column] = new_vx_star

            elif column == 5:
                # calculating vy star
                vy = my_array[row-1, 12]
                rx = my_array[row-1, 8]
                ry = my_array[row - 1, 9]
                rz = my_array[row - 1, 10]
                new_vy_star = calculate_vy_star(vy, dt, G, M, rx, ry, rz)
                my_array[row, column] = new_vy_star

            elif column == 6:
                # calculating vz star
                vz = my_array[row-1, 13]
                rx = my_array[row - 1, 8]
                ry = my_array[row - 1, 9]
                rz = my_array[row - 1, 10]
                new_vz_star = calculate_vz_star(vz, dt, G, M, rx, ry, rz)
                my_array[row, column] = new_vz_star

            elif column == 7:
                # calculating t
                rz = my_array[row-1, 10]
                new_t = calculate_t(rz, dt)
                my_array[row, column] = new_t

            elif column == 8:
                # calculating rx
                rx = my_array[row-1, 8]
                vx_star = my_array[row, 4]
                new_rx = calculate_rx(rx, vx_star, dt)
                my_array[row, column] = new_rx

            elif column == 9:
                # calculating ry
                ry = my_array[row-1, 9]
                vy_star = my_array[row, 5]
                new_ry = calculate_ry(ry, vy_star, dt)
                my_array[row, column] = new_ry

            elif column == 10:
                # calculating rz
                rz = my_array[row-1, 10]
                vz_star = my_array[row, 6]
                new_rz = calculate_rz(rz, vz_star, dt)
                my_array[row, column] = new_rz

            elif column == 11:
                # calculating vx
                vx = my_array[row-1, 11]
                rx_star = my_array[row, 1]
                ry_star = my_array[row, 2]
                rz_star = my_array[row, 3]

                new_vx = calculate_vx(vx, G, M, rx_star, ry_star, rz_star, dt)

                my_array[row, column] = new_vx

            elif column == 12:
                # calculating vy
                vy = my_array[row-1, 12]
                rx_star = my_array[row, 1]
                ry_star = my_array[row, 2]
                rz_star = my_array[row, 3]
                new_vy = calculate_vy(vy, G, M, rx_star, ry_star, rz_star, dt)
                my_array[row, column] = new_vy

            elif column == 13:
                # calculating vz
                vz = my_array[row - 1, 13]
                rx_star = my_array[row, 1]
                ry_star = my_array[row, 2]
                rz_star = my_array[row, 3]
                new_vy = calculate_vy(vy, G, M, rx_star, ry_star, rz_star, dt)
                new_vz = calculate_vz(vz, G, M, rx_star, ry_star, rz_star, dt)
                my_array[row, column] = new_vz

            else:
                print("Column not found")
    return my_array

def main():

    orbit_vectors = calculate_orbit(1200000)

    df = pd.DataFrame(orbit_vectors, columns=['t*', 'rx*', 'ry*', 'rz*', 'vx*','vy', 'vz*', 't', 'rx', 'ry', 'rz', 'vx', 'vy', 'vz'])
    print(df)

    # Saving to csv
    df.to_csv('parameters.csv')


main()
