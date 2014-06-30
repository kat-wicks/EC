#! /usr/bin/env python

# The MIT License (MIT)

# Copyright (c) 2014 Erik Hemberg

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
*Non-adversarial* version of **STU Tron**. (Tron is named after a game in the
movie Tron http://en.wikipedia.org/wiki/Tron )

.. _tron_non_adversarial_ex:
.. figure:: images/tron_non_adversarial.png
   :align: center
   :width: 120px
   :height: 120px
   :alt: Toroidal grid
   :figclass: align-center

   **STU Tron** Non-Adversarial, The player is squaring off with itself.

.. codeauthor:: Erik Hemberg <hembergerik@csail.mit.edu>

**STU Tron** Game description:
------------------------------

Components
~~~~~~~~~~

There are some components in the **STU Tron** game:

**Player**
  Controls the *direction* of a *Bike*, The **PlayerAI** controls the bike by
  using a predefined strategy.

  .. _directions:

  *Directions* are:

  - East -- (1, 0), 0 degrees
  - North -- (0, 1), 90 degrees
  - West -- (-1, 0), 180 degrees
  - South -- (0, -1), 270 degrees

  *Bike* properties:

  - Moves ahead in the current direction with constant speed.
  - Leaves a *trail* on each square that has been visited. The
    tron_non_adversarial_ex_ shows the bike and the trail on the board.

.. figure:: images/toroidal.png
   :align: right
   :width: 200px
   :height: 100px
   :alt: Toroidal grid
   :figclass: align-center

   The **board** is donut shaped, i.e. a toroidal grid.

**Board**
  A square toroidal grid with length *n*, i.e. donut shaped

  - Exit at the north of the board and enter at the south of the board
  - Exit at the south of the board and enter at the north of the board
  - Exit to the west of the board and enter to the east of the board
  - Exit to the east of the board and enter to the west of the board

**Tron**
  The game contains the logic, checks for I/O events and coordinates
  the components.

.. note:: The **STU Tron** can be displayed in a GUI by passing the `-d` flag
   when starting the game.

Rules
~~~~~

The rules of the game:

- Score is given by the length of the trail. The maximum score is *n* * *n*
- Game ends when the bike hits the trail

.. _actions:

Actions
~~~~~~~

The player can take three actions in order to change direction:

- Left -- '<' turn player 90 degrees to the left
- Right -- '>' turn player 90 degrees to the right
- Ahead -- do nothing, i.e. do not change direction

Running **STU Tron** non-adversarial
------------------------------------

::

  usage: tron_non_adversarial.py [-h] [-r ROWS] [-d] [-f FILE]

  optional arguments:
    -h, --help            show this help message and exit
    -r ROWS, --rows ROWS  # of board rows
    -d, --draw_board      draw the board or not
    -f FILE, --file FILE
                        ai player strategy file

**STU Tron** non-adversarial classes
------------------------------------
"""

from Tkinter import Canvas, Tk, TclError
import datetime
import time

import argparse


class Player(object):
    """
    Player controls the bike in the STU Tron game, i.e. it is the bike. The
    player can draw
    its trail on the canvas.

    The player update consists of:

    - move the bike forward by adding the direction to
      the current coordinate.
    - append the new coordinate to the trail
    - check if the player has collided with the trail

    The actions are described in: :ref:`actions`
    
    Attributes:
      - Position -- The position is an x,y coordinate. When the bike moves
        the position is updated based on the current direction.
      - Direction -- The direction that the bike is facing
      - Board -- A reference to the game board. Used to:

        - Update the trail
        - Check for collisions when the bike moves

      - Alive -- Indicate if the player is alive
      - ID -- Identifier for the player
      - Canvas -- A reference to the canvas. Used if the board is displayed
        in the GUI

    .. _player_directions:

    PLAYER_DIRECTIONS
      Directions are given as (x, y) coordinates in the unit circle.
      Possible directions for the player are given in:
      :ref:`Directions <directions>` Ordering is based on
      the TKinter coordinate system. This allows the player to be drawn
      correctly in the GUI

    """

    # Directions player can move in [x, y] coordinates.
    # order is EAST, NORTH, WEST, SOUTH
    PLAYER_DIRECTIONS = ((1, 0), (0, -1), (-1, 0), (0, 1))

    def __init__(self, x, y, direction, color, alive, id_, canvas, board):
        """
        Constructor

        Arguments:

        :param x: x coordinate
        :type x: int
        :param y: y coordinate
        :type y: int
        :param direction: coordinates for the current direction
        :type direction: tuple of integers [0, 1]
        :param color: color of the player
        :type color: str
        :param alive: player life indicator
        :type alive: bool
        :param id_: unique identifier
        :type id_ int
        :param canvas: canvas to draw player on
        :type canvas: TronCanvas
        :param board: game board
        :type board: list
        """
        # x coordinate
        self.x = x
        # y coordinate
        self.y = y
        # Bike direction
        self.direction = direction
        # Bike color
        self.color = color
        # Is player alive
        self.alive = alive
        # Player id
        self.ID = id_
        # Canvas to draw on
        self.canvas = canvas
        # Board to move on
        self.board = board
        # Store the board coordinates of the bike trail. The entire
        # bike needs to be redrawn each time
        self.bike_trail = []

    def move_bike(self):
        """
        Move the player(bike) by setting the coordinates based on
        the current coordinates and direction.
        """
        # Get new x coordinate given the current direction
        self.x = self.get_new_point(self.x, self.direction[0])
        # Get new y coordinate given the current direction
        self.y = self.get_new_point(self.y, self.direction[1])

    #        print("move_bike direction:(%s) coordinate:[%d, %d]" %
    #              (' '.join(map(str, self.direction)), self.x, self.y))

    def draw_bike(self):
        """
        Draw the coordinates of the bike trail on the canvas. Each
        part of the trail is a rectangle with the player's color.
        """
        # Draw each part of the bike trail

        for row, col in self.bike_trail[:]:
            #Create rectangle with coordinates x1, y1, x2, y2,
            #i.e. the top left and bottom right corner.

            self.canvas.tk_canvas.create_rectangle(
                row * self.canvas.bike_width,
                col * self.canvas.bike_height,
                (row + 1) * self.canvas.bike_width,
                (col + 1) * self.canvas.bike_height,
                fill=self.color)

    def update(self):
        """
        Update the player by:

        - move the player in the current direction
        - check for collision, i.e. board value is > 0.
        - mark the board with the current length of the trail at the current
          coordinates
        - append the current coordinates to the trail.

        """
        # Move the bike if alive

        if self.alive:
            self.move_bike()

        # Check if bike has crashed, i.e. the board cell at the
        # current coordinate is not 0

        if self.board[self.x][self.y] != 0:
            self.alive = False
        else:
            # Append the current coordinate to the bike trail

            self.bike_trail.append((self.x, self.y))

            # Set board at current coordinate to be non empty, i.e. not
            # 0. The size of the bike trail is used as the new cell
            # value

            self.board[self.x][self.y] = len(self.bike_trail)

    def ahead(self):
        """
        Do nothing. The direction does not need to be changed.
        """
        pass

    def left(self):
        """
        Change the direction by turning left 90 degrees.

        The turn is
        implemented by finding the index of the current direction and
        increasing it by one. The modulus is used ot get a valid index.
        """
        # Get the index of the current direction

        direction_index = Player.PLAYER_DIRECTIONS.index(self.direction)

        # Increase the index by one. Use the modulus to get a valid index

        new_direction_index = (direction_index + 1) % \
                              len(Player.PLAYER_DIRECTIONS)

        # Set the new direction

        self.direction = Player.PLAYER_DIRECTIONS[new_direction_index]

    def right(self):
        """
        Change the direction by turning right 90 degrees.

        The turn is
        implemented by finding the index of the current direction and
        decreasing it by one. The modulus is used ot get a valid index.
        """
        # Get the index of the current direction

        direction_index = Player.PLAYER_DIRECTIONS.index(self.direction)

        # Decrement index by 1, but we increment by length of
        # PLAYER_DIRECTIONS -1 and take the modulus to avoid getting -1
        # if the current index is 0

        new_direction_index = (direction_index +
                               len(Player.PLAYER_DIRECTIONS) - 1) % \
                              len(Player.PLAYER_DIRECTIONS)

        # Set the new direction

        self.direction = Player.PLAYER_DIRECTIONS[new_direction_index]

    def get_new_point(self, point, delta):
        """
        Returns an integer point based on the current point and
        delta. The board is toroidal.

        :param point: a point on the board
        :type point: int
        :param delta: the value that changes the point
        :type delta: int

        """
        # Get the new point. Use modulus since the board is
        # toroidal. Add the length of the board to avoid getting
        # negative values.

        point_p = (point + delta + len(self.board)) % len(self.board)

        return point_p


class PlayerAI(Player):
    """
    The AI player extends Player. The AI has a predefined strategy which is
    executed every step of the game.

    The AI player can sense its environment. It is from this knowledge of the
    environment that it decides which action to take. The sensors can report
    back if
    there is an obstacle ahead in the :ref:`Directions <directions>` which the
    bike can move.

    Attributes:
    - Strategy -- The strategy of the AI Player
    """

    def __init__(self, x, y, direction, color, alive, id_, canvas, board,
                 strategy):

        """
        Constructor. Sets the strategy of the AI player.

        :param x: x coordinate
        :type x: int
        :param y: y coordinate
        :type y: int
        :param direction: coordinates for the current direction
        :type direction: tuple
        :param color: color of the player
        :type color: str
        :param alive: player life indicator
        :type alive: bool
        :param id_: unique identifier
        :type id_ int
        :param canvas: canvas to draw player on
        :type canvas: TronCanvas
        :param board: game board
        :type board: list
        :param strategy: player strategy
        :type strategy: str
        """
        Player.__init__(self, x, y, direction, color, alive, id_, canvas, board)

        # The strategy of the AI. A strategy is a policy for what actions to
        # take given the sensed environment.

        self.strategy = strategy

        # The environment that the AI senses

        self.environment = {}

    def evaluate_strategy(self):
        """
        Evaluate AI Player strategy. First check the current
        environment. Evaluate the strategy based on the current
        environment.
        """
        # Check the environment of the Player
        self.check_environment()
        # Execute the strategy, pass in self as the environment for execution
        exec (self.strategy, {"self": self})

    def check_environment(self):
        """
        Find distance to obstacles(bike trails) in
        :ref:`Directions <directions>` of the current coordinates. Distance is
        measured in number of squares.
        """
        # Clear the environment

        self.environment = {}

        # Get the directions of the adjacent cells

        for adjacent_cell in Player.PLAYER_DIRECTIONS:

            # Get coordinates of the adjacent cell

            _x = self.get_new_point(self.x, adjacent_cell[0])
            _y = self.get_new_point(self.y, adjacent_cell[1])

            # Distance to obstacles

            distance = 0

            # How far is it to an obstacle. The max distance is the
            # length of the board

            while self.board[_x][_y] == 0 and distance < len(self.board):
                # Increase distance

                distance += 1

                # Get coordinates of the adjacent cell

                _x = self.get_new_point(_x, adjacent_cell[0])
                _y = self.get_new_point(_y, adjacent_cell[1])

            # Set the distance to and obstacle in the direction in the
            # environment

            self.environment[adjacent_cell] = distance

    def is_obstacle_in_relative_direction(self, direction):
        """
        Return a boolean denoting if the distance to an obstacle in the
        environment in the relative direction is one cell ahead.

        :param direction: relative direction to look in
        :type direction: int
        :returns: if obstacle is ahead
        :rtype: bool

        """
        # Threshold for how far ahead an obstacle is reported
        threshold = 1.0 / float(len(self.board))
        # Distance to obstacle
        distance = self.distance(direction)
        return distance < threshold

    def distance(self, direction):
        """
        Return a float [0, 1] that is the distance in the
        environment in the direction relative to the player direction
        divided by the board length.

        :param direction: relative direction to look in
        :type direction: int
        :returns: distance to obstacle [0,1]
        :rtype: float
        """
        direction_index = Player.PLAYER_DIRECTIONS.index(self.direction)
        new_direction_index = (direction_index + len(Player.PLAYER_DIRECTIONS) +
                               direction) % len(Player.PLAYER_DIRECTIONS)
        new_direction = Player.PLAYER_DIRECTIONS[new_direction_index]
        dist = float(self.environment[new_direction]) / float(len(self.board))
        return dist


class TronCanvas(object):
    """
    Tron canvas class, contains the fields related to drawing the game. Canvas
    is used to draw the Players on the screen. The drawing is done by Tk.

    Attributes:

    - Root -- The root Tk instance
    - Canvas -- The TkCanvas
    - Bike width -- Width of the bike in pixels
    - Rows -- Number of cells in each row of the board.
      The Bike width x Rows + 2 x MARGIN gives the size in pixels of the
      canvas.

    MARGIN
      The margin of the canvas

    DELAY
      Delay time between redrawing the canvas

    GUI_DISPLAY_AFTER_COLLISION
      Time in seconds GUI is displayed after a collision. Default is 2 seconds
    """

    # Margin of canvas in pixels
    MARGIN = 5
    # Delay between updates in milliseconds
    DELAY = 150
    # Time GUI is displayed after a collision
    GUI_DISPLAY_AFTER_COLLISION = 2

    def __init__(self, rows, bike_width):
        """
        Constructor

        :param rows: number of rows on the grid
        :type rows: int
        :param bike_width: width of bike when drawing
        :type bike_width: int
        """
        # Create Tk instance
        self.root = Tk()
        # Set window to be resizable
        self.root.resizable(width=0, height=0)
        # Canvas to draw on
        canvas_width = 2 * TronCanvas.MARGIN + rows * bike_width
        canvas_height = canvas_width
        # Create a Tk Canvas instance
        self.tk_canvas = Canvas(self.root, width=canvas_width,
                                height=canvas_height)
        # Geometry manager organizes widgets in blocks before placing
        # them in the parent widget
        self.tk_canvas.pack()
        # Set bike width
        self.bike_width = bike_width
        # Set bike height
        self.bike_height = bike_width
        # Set number of rows
        self.rows = rows


class Tron(object):
    """
    Tron represents the game, it has:

     - a board, which is a square toroidal grid
     - a canvas which is used when drawing and displaying the player
     - a player which controls the bike that is constantly moving forward and
       leaves a trail

    **STU Tron** game flow:

    *Initialization*
      The board, player and canvas are initialized. The listeners of the
      canvas are initialized

    *Run*
      Update the player each step of the game. The direction of the player is
      updated by key events. The key events trigger an :ref:`actions` of a
      player:

       - A human player presses the '<' or '>' arrow on the keyboard. The
         direction of the player is changed and the canvas is redrawn.
       - An AI player executes a strategy

    *Cleanup*
      The game ends when the player collides with and obstacle. Close the GUI
      and write the statistics of the game to a log file.

    Attributes:

    - Canvas -- Used for displaying the GUI
    - Rows -- The number of cells on the board
    - Board -- The board where player moves. Keeps track of the trail that
      the bike leaves behind
    - Bike Width -- The width in pixels of the bike. Used when drawing the
      bike on the GUI
    - Draw board -- Indicator for turning the GUI on or off.
    - Player -- The player controlling the bike. The initial position of the
      player is at the center of the board and the initial direction is facing
      south
    - Strategy -- The predefined strategy of a player, if the strategy is
      `None` a human can control the player

    .. note:: Executing **STU Tron** without a GUI is faster. There is no
       need to draw on the canvas and DELAY each step.

    STATS_FILE
      Name of file for saving the statistics from each game

    """

    STATS_FILE = "tron_game_stats.log"

    def __init__(self, rows, bike_width, draw_board, strategy):
        """
        Constructor

        :param rows: number of rows on the grid
        :type rows: int
        :param bike_width: width of bike when drawing
        :type bike_width: int
        :param draw_board: draw bikes on the board
        :type draw_board: bool
        :param strategy: predefined strategy of player
        :type strategy: str
        """
        # Canvas that the GUI can draw on
        self.canvas = None
        self.draw_board = draw_board
        # Check if we are drawing the board
        if self.draw_board:
            # Create a canvas to draw on

            self.canvas = TronCanvas(rows=rows, bike_width=bike_width)

        self.game_over = False
        # The board is a matrix of size row x row. 0 indicates that
        # the cell is unoccupied
        self.board = []
        for i in range(rows):

            row = []
            for j in range(rows):
                row.append(0)

            self.board.append(row)

        # Create human or AI player if a strategy argument is passed in
        if not strategy:

            self.player = Player(x=rows / 2, y=rows / 2, direction=(0, 1),
                                 color="Blue", alive=True, id_=0,
                                 canvas=self.canvas, board=self.board)

        else:
            self.player = PlayerAI(x=rows / 2 + 1, y=rows / 2, direction=(0, 1),
                                   color="Green", alive=True, id_=1,
                                   canvas=self.canvas, board=self.board,
                                   strategy=strategy)

        self.winner = None

    def step(self):
        """
        Function called if GUI not initialized. Performs a step of the
        game:

         - Updates the player
         - Check if the game is over.
        """
        # Make the move of the AI player
        if isinstance(self.player, PlayerAI):
            self.ai_key_pressed(self.player)

        if not self.game_over:

            # Update the player
            self.player.update()
            # Check if player is alive
            if not self.player.alive:
                self.game_over = True
                self.write_stats()

    def step_and_draw(self):
        """
        A step of the game:

         - Updates the player
         - Redraws the board
         - Calls itself after a DELAY.
         - Destroys the GUI if game is over

        """
        if self.game_over:
            # Wait
            time.sleep(TronCanvas.GUI_DISPLAY_AFTER_COLLISION)
            # Destroy the canvas. This closes the window
            try:
                self.canvas.root.destroy()
            except TclError:
                return
        self.step()

        # Redraw the board
        self.redraw_all()

        # Pause, then call step_and_draw again
        self.canvas.tk_canvas.after(TronCanvas.DELAY, self.step_and_draw)

    def key_pressed(self, event):
        """
        Determine the action when a key is pressed.

        Key events:

        - Left arrow *<* turns player 90 degrees counter clockwise
        - Right arrow *>* key turns the player 90 degrees clockwise
        - *q* sets the game to be over

        :param event: event on GUI
        :type event: TkInterEvent
        """
        # Process keys that work even if the game is over
        if event.char == "q":
            self.game_over = True

        # Process keys that only work if the game is not over
        if not self.game_over:

            if event.keysym == "Left":

                self.player.left()

            elif event.keysym == "Right":

                self.player.right()

            # Redraw the board
            if self.draw_board:
                self.redraw_all()

    def ai_key_pressed(self, player):
        """Evaluate the strategy of the player.

        :param player: player strategy to evaluate
        :type player: PlayerAI
        """
        if not self.game_over:
            # Evaluate the strategy of the player
            player.evaluate_strategy()

    def write_stats(self):
        """
        Write the statistics of the game to a file when the game is over.

        The statistics recorded are:

        - Time of game
        - Number of rows on the board
        - ID of player
        - Length of player trail
        - Strategy used by player. (A human player strategy is *None*)

        >>>python tron_non_adversarial.py
        Time; 2014-02-25 09:54:50.871124
        STU Tron Non-Adversarial; board size 4; bike trail length 4
        GUI; False
        Trail; [(2, 3), (2, 0), (2, 1), (2, 2)]

        """
        if self.game_over:

            # Set the size of the bike trail
            text_str = 'board rows %d; bike trail length %s' % \
                       (len(self.board), len(self.player.bike_trail))
            time_stamp = datetime.datetime.now()
            strategy = None
            if isinstance(self.player, PlayerAI):
                strategy = self.player.strategy

            print('Time; %s' % time_stamp)
            print('STU Tron Non-Adversarial; %s' % text_str)
            print('GUI; %s' % self.draw_board)
            print('Trail; %s' % self.player.bike_trail)
            print('Strategy;\n%s' % str(strategy))

            # Write stats to file
            f_out = open(Tron.STATS_FILE, 'a')
            # Write, time, rows, id, bike trail length, strategy
            f_out.write('%s, %d, %d, %d, strategy:\n%s\n' %
                        (time_stamp, len(self.board), self.player.ID,
                         len(self.player.bike_trail), str(strategy)))
            f_out.close()

    def redraw_all(self):
        """
        Redraw the canvas:

         - The bike is redrawn.
         - If game is over:

           - the statistics are written to GUI

        """
        try:

            # Clear all elements
            self.canvas.tk_canvas.delete("all")
            # Draw the bike of the player
            self.player.draw_bike()
            # Check if game is over
            if self.game_over:
                # Display the winner on the board
                text_str = 'Trail length:%d' % len(self.player.bike_trail)
                self.canvas.tk_canvas.create_text(
                    100, 10, text=text_str,
                    font=("Helvetica", 12, "bold"))

        except TclError:

            pass

    def run(self):
        """
        Run the Tron game:

        GUI
          When the GUI is displayed the direction keys are
          bound to the canvas and each step of the game will redraw the GUI.

        No GUI
          GUI is not invoked. The execution is *faster*.
        """

        # Check if the GUI will be used to draw the board
        if self.draw_board:

            # Draw initial board
            self.redraw_all()
            # Set up events and bind keys to the events
            self.canvas.root.bind("<Key>", self.key_pressed)
            # Call game step_and_draw, the function calls itself after DELAY
            # time
            self.step_and_draw()
            # Launch the GUI. This call BLOCKS
            self.canvas.root.mainloop()

        else:

            # Step the game forward as long as it has not ended
            while not self.game_over:
                self.step()


def read_strategy_from_file(file_name):
    """
    Read a strategy from a file. Return the strategy as a string
    
    :param file_name:
    :type file_name: str
    :returns: strategy 
    :rtype: str
    """
    in_file = open(file_name, 'r')
    strategy = in_file.read()
    in_file.close()
    return strategy


def main():
    """
    Setup and run the tron game.

    - Read the command-line arguments.
    - Create **STU Tron** objects
    - Run the game
    """

    # If args are specified, set them
    parser = argparse.ArgumentParser()
    # Number of rows on the board
    parser.add_argument("-r", "--rows", type=int, default=40,
                        help="# of board rows")
    # Draw the board
    parser.add_argument("-d", "--draw_board", action='store_false',
                        help="draw the board or not")
    # File with an AI Player strategy
    parser.add_argument("-f", "--file", default="",
                        help="ai player strategy file")

    # Read the commandline arguments
    args = parser.parse_args()
    draw_board = args.draw_board
    rows = args.rows
    player_strategy = None
    if args.file:
        player_strategy = read_strategy_from_file(args.file)

    bike_width = 4

    # Create the game
    tron = Tron(rows=rows, bike_width=bike_width, draw_board=draw_board,
                strategy=player_strategy)
    # Run the game
    tron.run()

# Entry point for standalone application, e.g. when run from the command-line
if __name__ == '__main__':
    main()
