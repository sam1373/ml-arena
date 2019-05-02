#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version by Ken Lauer / sirkne at gmail dot com
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

# Original C++ version by Daid
#  http://www.box2d.org/forum/viewtopic.php?f=3&t=1473
# - Written for pybox2d 2.1 by Ken
import sys

from .framework import (Framework, Keys, main)
from Box2D import (b2AssertException, b2Color, b2EdgeShape, b2FixtureDef,
                   b2PolygonShape, b2CircleShape, b2RayCastCallback, b2Vec2, b2ChainShape,
                   b2Mul, b2World)

import random
import math

from .enemy import *

import pygame
from pygame.surfarray import *
from pygame.locals import *
from numpy import *

from .agent import *
from .nn import *
from .bullet import Bullet

import numpy as np

from copy import deepcopy

from skimage.transform import resize

import matplotlib.pyplot as plt

LASER_HALF_WIDTH = 1
LASER_SPLIT_SIZE = 0.01
LASER_SPLIT_TAG = 'can_cut'
BLOOD_TAG = 'is_blood'
DANGER = 'DANGER'
PLAYER_PART = 'pp'

NUM_SIGHT_RAYS = 15
SIGHT_ANGLE = 3.1415 * 0.2

class RayCastClosestCallback(b2RayCastCallback):
    """This callback finds the closest hit"""

    def __repr__(self):
        return 'Closest hit'

    def __init__(self, ignore=[], **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False
        self.ignore = ignore

    def ReportFixture(self, fixture, point, normal, fraction):
        '''
        Called for each fixture found in the query. You control how the ray
        proceeds by returning a float that indicates the fractional length of
        the ray. By returning 0, you set the ray length to zero. By returning
        the current fraction, you proceed to find the closest point. By
        returning 1, you continue with the original ray clipping. By returning
        -1, you will filter out the current fixture (the ray will not hit it).
        '''
        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        # NOTE: You will get this error:
        #   "TypeError: Swig director type mismatch in output value of
        #    type 'float32'"
        # without returning a value
        if self.fixture.body in self.ignore:
            return -1
        return fraction

def get_mass_center_shape(sh):
    p = b2Vec2(0, 0)
    for i in sh.vertices:
        p += i
    p /= len(sh.vertices)
    return p

def get_mass_center(obj):
    p = b2Vec2(0, 0)
    tr = obj.transform
    if len(obj.fixtures) == 0:
        return obj.position
    if len(obj.fixtures[0].shape.vertices) == 0:
        return obj.position
    #print(obj.fixtures[0].shape.vertices)
    for i in obj.fixtures[0].shape.vertices:
        p += b2Mul(tr, i)
    p /= len(obj.fixtures[0].shape.vertices)
    return p

def has_sight(world, agent, obj):
    if agent == obj:
        return 1
    callback = RayCastClosestCallback(ignore=[agent])
    objMassPos = get_mass_center(obj)
    world.RayCast(callback, agent.position, objMassPos)
    if callback.hit:
        if callback.fixture.body == obj:
            return callback.point
    return None

def get_sight(world, agent, targetPos):
    callback = RayCastClosestCallback(ignore=[agent])
    world.RayCast(callback, agent.position, targetPos)
    if callback.hit:
        return callback.point, callback.fixture.body
    return None

def get_poly(n_vert=5, scale=1, angPlus=0):
    verts = []
    for i in range(n_vert):
        ang = 2 * 3.1415 / n_vert * i
        vert = b2Vec2(math.cos(ang + angPlus), math.sin(ang + angPlus)) * scale
        verts.append(vert)

    return b2PolygonShape(vertices=verts)

def get_random_vector(scale=1):
    return b2Vec2(random.uniform(-1, 1), random.uniform(-1, 1)) * scale

def body_explode(world, body, p, force=1, userData=DANGER):
    if(len(body.fixtures) == 0):
        print("body has no fixtures")
        return
    print("exploded")
    fixture = body.fixtures[0]
    polygon = fixture.shape

    vol = body.mass / fixture.density

    if vol < 0.3:
        print("small mass")
        world.DestroyBody(body)
        return

    pBody = body.GetLocalPoint(p)

    verts = polygon.vertices
    n = len(verts)
    print(n)

    for i in range(len(verts)):
        if pBody == verts[i] or pBody == verts[(i + 1) % n]:
            pBody = get_mass_center(body)
            pBody = body.GetLocalPoint(pBody)
        new_shape_verts = [verts[i], verts[(i + 1) % n], pBody]

        new_shape = b2PolygonShape(vertices=new_shape_verts)

        new_body = world.CreateDynamicBody(
            userData=userData,
            position=body.position,
            angle=body.angle,
            linearVelocity=body.linearVelocity,
            angularVelocity=body.angularVelocity,
            linearDamping=1,
            angularDamping=1,
        )

        new_body.CreateFixture(
            friction=fixture.friction,
            restitution=0.8,#fixture.restitution,
            density=fixture.density,
            shape=new_shape,
        )

        #print("new mass:", new_body.mass)
        vol = new_body.mass / fixture.density

        if vol < 0.3:
            print("small mass")
            world.DestroyBody(new_body)
            continue

        rand_vec = get_random_vector(1000) * force#b2Vec2(random.uniform(-1, 1), random.uniform(-1, 1)) * 1000

        new_body.ApplyLinearImpulse(rand_vec, body.position, True)

    print("explode 2")

    world.DestroyBody(body)

    print("explode 3")

def _polygon_split(fixture, p1, p2, split_size):
    polygon = fixture.shape
    body = fixture.body
    # transform = body.transform

    local_entry = body.GetLocalPoint(p1)
    local_exit = body.GetLocalPoint(p2)
    entry_vector = local_exit - local_entry
    entry_normal = entry_vector.cross(1.0)
    last_verts = None
    new_vertices = [[], []]
    cut_added = [-1, -1]
    for vertex in polygon.vertices:
        # Find out if this vertex is on the new or old shape
        if entry_normal.dot(b2Vec2(vertex) - local_entry) > 0.0:
            verts = new_vertices[0]
        else:
            verts = new_vertices[1]

        if last_verts != verts:
            # if we switch from one shape to the other, add the cut vertices
            if last_verts == new_vertices[0]:
                if cut_added[0] != -1:
                    return []
                cut_added[0] = len(last_verts)
                last_verts.append(b2Vec2(local_exit))
                last_verts.append(b2Vec2(local_entry))
            elif last_verts == new_vertices[1]:
                if cut_added[1] != -1:
                    return []
                cut_added[1] = len(last_verts)
                last_verts.append(b2Vec2(local_entry))
                last_verts.append(b2Vec2(local_exit))

        verts.append(b2Vec2(vertex))
        last_verts = verts

    # Add the cut if not added yet
    if cut_added[0] < 0:
        cut_added[0] = len(new_vertices[0])
        new_vertices[0].append(b2Vec2(local_exit))
        new_vertices[0].append(b2Vec2(local_entry))
    if cut_added[1] < 0:
        cut_added[1] = len(new_vertices[1])
        new_vertices[1].append(b2Vec2(local_entry))
        new_vertices[1].append(b2Vec2(local_exit))

    # Cut based on the split size
    for added, verts in zip(cut_added, new_vertices):
        if added > 0:
            offset = verts[added - 1] - verts[added]
        else:
            offset = verts[-1] - verts[0]
        offset.Normalize()
        verts[added] += split_size * offset

        if added < len(verts) - 2:
            offset = verts[added + 2] - verts[added + 1]
        else:
            offset = verts[0] - verts[len(verts) - 1]
        offset.Normalize()
        verts[added + 1] += split_size * offset

    # Ensure the new shapes aren't too small
    for verts in new_vertices:
        for i, v1 in enumerate(verts):
            for j, v2 in enumerate(verts):
                if i != j and (v1 - v2).length < 0.1:
                    # print('Failed to split: too small')
                    return []

    try:
        #for i in range(len(new_vertices)):
        #    if len(new_vertices[i])
        #apparently can't have more than 16
        return [b2PolygonShape(vertices=verts[:16]) for verts in new_vertices]
    except b2AssertException:
        return []
    except ValueError:
        return []


def _laser_cut(world, start_pos, laser_dir, length=30.0, laser_half_width=2, **kwargs):
    p1, p2 = get_laser_line(start_pos, laser_dir, length, laser_half_width)

    callback = laser_callback()
    world.RayCast(callback, p1, p2)
    if not callback.hit:
        return []
    hits_forward = callback.hits

    callback = laser_callback()
    world.RayCast(callback, p2, p1)
    if not callback.hit:
        return []

    hits_reverse = callback.hits

    if len(hits_forward) != len(hits_reverse):
        print("uh what")
        return []

    ret = []
    for (fixture1, point1), (fixture2, point2) in zip(hits_forward, hits_reverse):
        # renderer.DrawPoint(renderer.to_screen(point1), 2, b2Color(1,0,0))
        # renderer.DrawPoint(renderer.to_screen(point2), 2, b2Color(0,1,0))
        # renderer.DrawSegment(renderer.to_screen(point1), renderer.to_screen(point2), b2Color(0,1,1))
        if fixture1 != fixture2:
            continue

        new_polygons = _polygon_split(
            fixture1, point1, point2, LASER_SPLIT_SIZE)
        if new_polygons:
            ret.append((fixture1, new_polygons))

    return ret

def createBloodPart(world, pos):
        size = 0.1 * random.uniform(0.8, 1.2)
        dir = random.random() * 2 * 3.1415
        rayDir = b2Vec2(math.sin(dir), math.cos(dir))
        blastPower = (random.random() + 0.5) * 30


        part = world.CreateDynamicBody(
                position=pos,
                userData=BLOOD_TAG,
                #bullet=True,
                linearVelocity = blastPower * rayDir,
                linearDamping=1,
                fixtures=b2FixtureDef(
                    density=10,
                    #friction=100,
                    shape=b2PolygonShape(box=(size, size)),
                    #friction=0,
                    restitution=0.9,
                    #groupIndex=-1,
                )
            )
        """
        float angle = (i / (float)numRays) * 360 * DEGTORAD;
          b2Vec2 rayDir( sinf(angle), cosf(angle) );
      
          b2BodyDef bd;
          bd.type = b2_dynamicBody;
          bd.fixedRotation = true; // rotation not necessary
          bd.bullet = true; // prevent tunneling at high speed
          bd.linearDamping = 10; // drag due to moving through air
          bd.gravityScale = 0; // ignore gravity
          bd.position = center; // start at blast center
          bd.linearVelocity = blastPower * rayDir;
          b2Body* body = m_world->CreateBody( &bd );
      
          b2CircleShape circleShape;
          circleShape.m_radius = 0.05; // very small
      
          b2FixtureDef fd;
          fd.shape = &circleShape;
          fd.density = 60 / (float)numRays; // very high - shared across all particles
          fd.friction = 0; // friction not necessary
          fd.restitution = 0.99f; // high restitution to reflect off obstacles
          fd.filter.groupIndex = -1; // particles should not collide with each other
          body->CreateFixture( &fd );
        """


def laser_cut(world, start_pos, laser_dir, length=30.0, laser_half_width=2, **kwargs):
    #laser_dir /= laser_dir.length
    laser_dir_90 = b2Vec2(-laser_dir[1], laser_dir[0])
    #print(laser_dir_90)


    cut_fixtures = _laser_cut(
        world, start_pos, laser_dir, laser_half_width=LASER_HALF_WIDTH)
    remove_bodies = []
    retBody = None
    for fixture, new_shapes in cut_fixtures:
        body = fixture.body
        if body in remove_bodies:
            continue

        new_body = world.CreateDynamicBody(
            userData=PLAYER_PART,
            #userData=LASER_SPLIT_TAG,
            position=body.position,
            angle=body.angle,
            linearVelocity=body.linearVelocity,
            angularVelocity=body.angularVelocity,
            linearDamping=1,
            angularDamping=1,
        )

        try:

            cen = get_mass_center_shape(new_shapes[1])
            new_verts = []
            for i in range(len(new_shapes[1].vertices)):
                ver = new_shapes[1].vertices[i]
                ver = b2Vec2(ver)
                new_verts.append(ver - cen)
            new_shapes[1].vertices = new_verts

            new_body.position += cen


            new_body.CreateFixture(
                friction=fixture.friction,
                restitution=0.7,#fixture.restitution,
                density=fixture.density,
                shape=new_shapes[1],
            )

            #impulse test
            m = new_body.mass
            m1 = body.mass
            if m1 < m:
                new_body, body = body, new_body
                m, m1 = m1, m
            m *= 18
            print(m)
            new_body.ApplyLinearImpulse(m * laser_dir_90, new_body.position, True)
            """
            bp = (m / m1) / 0.1# + random.randint(1, 3)
            bp = int(bp)
            print(m, bp)
            for i in range(bp):
                createBloodPart(world, (body.position + new_body.position) / 2.0)
            """
            #else:
            #body.ApplyLinearImpulse(-m * 0.3 * laser_dir_90, body.position, True)
            retBody = new_body

        except AssertionError:
            print('New body fixture failed: %s' % sys.exc_info()[1])
            remove_bodies.append(new_body)

        try:
            body.CreateFixture(
                friction=fixture.friction,
                restitution=fixture.restitution,
                density=fixture.density,
                shape=new_shapes[0],
            )

            body.DestroyFixture(fixture)
        except AssertionError:
            print('New fixture/destroy failed: %s' % sys.exc_info()[1])
            remove_bodies.append(body)

    for body in remove_bodies:
        world.DestroyBody(body)

    print("done laser")

    return retBody



def get_laser_line(start_pos, laser_dir, length, laser_half_width):
    #laser_start = (laser_half_width - 0.1, 0.0)
    #laser_dir = (length, 0.0)
    p1 = start_pos#laser_body.GetWorldPoint(laser_start)
    p2 = p1 + laser_dir#laser_body.GetWorldVector(laser_dir)
    return (p1, p2)


def laser_display(renderer, start_pos, laser_dir, length=30.0, laser_color=(1, 1, 1), laser_half_width=2, **kwargs):
    if not renderer:
        return

    p1, p2 = get_laser_line(start_pos, laser_dir, length, laser_half_width)
    renderer.DrawSegment(renderer.to_screen(
        p1), renderer.to_screen(p2), b2Color(*laser_color))



class laser_callback(b2RayCastCallback):
    """This raycast collects multiple hits."""

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.hit = False
        self.hits = []

    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True

        if fixture.body.userData == LASER_SPLIT_TAG:
            self.hits.append((fixture, point))

        self.last_fixture = fixture
        self.last_point = point
        return 1.0

def getRGB(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return red, green, blue

def get_sight_ray(world, agent, ang):
    targetPos = agent.position + b2Vec2(cos(ang) * 500, sin(ang) * 500)
    callback = RayCastClosestCallback(ignore=[agent])
    world.RayCast(callback, agent.position, targetPos)
    if callback.hit:
        return callback.point, callback.fixture.body
    return None, None


class Arena(Framework):
    name = "ML Arena"
    #description = 'Press (c) to cut'
    move = 0
    jump = 100
    #startMouse = (0, 0)
    #mouseDown = 0

    def __init__(self):
        super(Arena, self).__init__()

        self.dark = 0
        self.glitch = 0
        self.noClear = 0
        self.glow = 0

        self.drawRays = 1

        

        self.mouseDown = 0
        self.mouseStart = (0, 0)
        self.mouseHeld = 0
        self.pressed = dict()
        for key in [Keys.K_w, Keys.K_a, Keys.K_s, Keys.K_d, Keys.K_q]:
            self.pressed[key] = False

        pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
        self.myfont = pygame.font.SysFont('Comic Sans MS', 20)

        self.nAgents = 10

        self.agents = [None for i in range(self.nAgents)]

        self.setLevel(1)
        self.clearScreen()

        
        

    #def beginContact(self, contact):
    #    print(contact)

    def resetLevel(self):
        self.setLevel(self.level)
        self.clearScreen()


    def setLevel(self, n=1):

        self.level = n


        self.setZoom(7)
        self.setCenter((-7, 8))

        #self.setZoom(3)
        #self.setCenter((63, -30))

        #self.viewCenter = b2Vec2(19, -10)#(b2Vec2(-7, 8))

        #self.viewZoom = 3#7

        prevAgentData = [None for i in range(self.nAgents)]

        parents = []

        for i, ag in enumerate(self.agents):
            if ag is not None:
                prevAgentData[i] = ag.userData
                if i != 0:
                    parents.append(i)


        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.world.destructionListener = self.destructionListener
        self.world.contactListener = self
        self.world.warmStarting = True
        self.world.continuousPhysics = True
        self.world.subStepping = False
        #for i in self.world.bodies:
        #    self.world.DestroyBody(i)


        playerPos = (-20, 0)

        self.exitPos = (20, 0)

        self.dark = self.glitch = 0


        if n == 1:
            left = -40
            right = 40
            down = -30
            up = 30
            # The ground
            self.ground = self.world.CreateStaticBody(
                userData='ground',
                shapes=[
                    b2EdgeShape(vertices=[(left, down), (right, down)]),
                    b2EdgeShape(vertices=[(left, down), (left, up)]),
                    b2EdgeShape(vertices=[(right, down), (right, up)]),
                    b2EdgeShape(vertices=[(left, up), (right, up)]),
                ]
            )

   



            



        plShape = get_poly(5, 5)#b2PolygonShape(vertices=self.getHexagonVertices(0.3))

        cols = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0.5, 0.5)]
         
        nAgents = self.nAgents

        #self.agents = []   

        for i in range(nAgents):

            if prevAgentData[i] is not None:
                agentData = prevAgentData[i]

                agentData.col *= 0.95

            else:
                if i != 0:
                    #if len(parents) == 0:
                    agentData = NeuralNetAgent(NN(NUM_SIGHT_RAYS * 4 * 2 + 5))

                    if len(parents) != 0:
                        r = random.choice(parents)
                        #agentData = deepcopy(prevAgentData[r])
                        w, b = prevAgentData[r].net.get_param()
                        agentData.net.set_param(w, b)
                        agentData.net.mutate()
                else:
                    agentData = PlayerAgent()

                agentData.col = b2Color(*cols[(i % len(cols))])

            agentData.id = i

            ang = pi * 2 / nAgents * i

            #pos = (10 + left + (right - 20 - left) / (nAgents - 1) * i, down + 10)
            pos = (cos(ang) * 20, sin(ang) * 20)


            newAgent = self.createAgent(agentData, pos)

            newAgent.angle = ang + 3.1415
            newAgent.userData.col = b2Color(*cols[(i % len(cols))])

            self.agents[i] = newAgent

        

        self.player = self.agents[0]

        

        self.using_contacts = True

        self.timer = 0

        self.game_over = 0

        #totalSpace = (up - down) * (right - left)
        #self.viewZoom = 3# - totalSpace / 6000.

        self.exitPos = b2Vec2(self.exitPos)




    def createAgent(self, agent, pos):
        newShape = get_poly(5, 1.4)

        newAgent = self.world.CreateDynamicBody(
                userData=agent,
                position=pos,
                linearDamping=3,
                angularDamping=1,
                fixtures=b2FixtureDef(
                    density=5.0,
                    restitution=0.8,
                    shape=newShape,
                )
            )

        return newAgent


    def getCutLine(self):
        #md = min(299, self.mouseHeld) / 300.0
        md = 0.08
        if self.mouseHeld > 100:
            md = 0.15
        if self.mouseHeld > 200:
            md = 0.25
        #prevents division by zero


        m = self.mouseWorld
        plPos = self.player.position

        #RayCastClosestCallback callback;//basic callback to record body and hit point
        #m_world->RayCast(&callback, center, rayEnd);
        #if ( callback.m_body ) 
        #  applyBlastImpulse(callback.body, center, callback.point, (m_blastPower / (float)numRays));
        callback = RayCastClosestCallback()
        plToM = m - plPos
        plToM /= plToM.length
        plToM *= 3.6
        self.world.RayCast(callback, plPos + plToM , plPos)
        if callback.hit:
            plToM = callback.point - plPos
        #plToM = (m - plPos)


        #plToM /= plToM.length
        plToM *= (0.9 - md)
        cutCenter = plPos + plToM
        plToM_90 = b2Vec2(-plToM[1], plToM[0])
        plToM_90 *= 1.0 / plToM_90.length * 5
        cut1 = cutCenter + plToM_90
        cut2 = cutCenter - plToM_90
        return cut1, cut2 - cut1
        return self.startMouse, self.mouseWorld - self.startMouse

    def isEnemy(self, obj):
        return issubclass(type((obj.userData)), Enemy)

    def Keyboard(self, key):
        self.pressed[key] = True
        #print(key)
        if key==Keys.K_e:
            for i in self.world.bodies:
                if self.isEnemy(i):
                    body_explode(self.world, i, i.position + get_random_vector(2), 3, DANGER)
            #body_explode(self.world, self.player, self.player.position + get_random_vector(2), 3)
        #if key==Keys.K_b:
        #    self.createBomber(self.mouseWorld)
        #if key==Keys.K_h:
        #    self.createAttractor(self.mouseWorld)
        if key==Keys.K_n:
            self.setLevel(self.level + 1)
        if key==Keys.K_p:
            self.setLevel(self.level - 1)
        if key==Keys.K_r:
            self.resetLevel()
        if key==Keys.K_f:
            self.drawRays = not self.drawRays

    def KeyboardUp(self, key):
        self.pressed[key] = False

    def MouseDown(self, p):
        #super(BoxCutter, self).MouseDown(p)
        if self.mouseDown == 0:
            self.startMouse = p
        self.mouseDown = 1

        self.agentShoot(self.player, self.mouseWorld)

    def MouseUp(self, p):
        #super(BoxCutter, self).MouseUp(p)
        if self.player is None:
            return
        print("click")
        
        self.mouseHeld = 0

    def MouseDownRight(self, p):
        return
        #cut_start, cut_dir = self.getCutLine()
        #new_body = laser_cut(self.world, cut_start, cut_dir,
        #              laser_half_width=LASER_HALF_WIDTH)

    def MouseMove(self, p):
        super(Arena, self).MouseMove(p)

    def enemyAttack(self, enemy, target):
        targetVec = target - enemy.position
        targetVec /= targetVec.length
        targetVec *= 100000
        enemy.ApplyLinearImpulse(targetVec, enemy.position, True)

    def agentShoot(self, agent, target):
        new_shape = get_poly(4, 0.6)
        targetVec = target - agent.position
        targetVec /= targetVec.length
        targetVec *= 2.
        bullet = self.world.CreateDynamicBody(
            userData=DANGER,
            position=(agent.position + targetVec),
            linearDamping=1,
            angularDamping=1,
            fixtures=b2FixtureDef(
                density=5.0,
                restitution=0.8,
                shape=new_shape,
            )
        )
        bullet.ApplyLinearImpulse(targetVec * 300, bullet.position, True)

    def isNoticable(self, body):
        vel = body.linearVelocity.length
        vel = min(vel, 15.)
        vel /= 15.0
        if vel > 0.5:
            return 1
        return 0

    def getDistance(self, pos1, pos2):
        return (b2Vec2(pos2) - b2Vec2(pos1)).length

    def drawBody(self, body, col, surface=None, scale=1., alpha=255):
        if surface is None:
            surface = self.renderer.surface
        sh = body.fixtures[0].shape
        verts = list(sh.vertices)
        for k in range(len(verts)):
            verts[k] = b2Mul(body.transform, b2Vec2(verts[k]) * scale)
        verts = list(map(self.renderer.to_screen, verts))
        #print(col.bytes)
        col = col.bytes + [alpha]
        pygame.draw.polygon(surface, col, verts, 0)

    def drawSolidPolygon(self, verts, col, pos=b2Vec2(0., 0.), surface=None):
        if surface is None:
            surface = self.renderer.surface
        verts = list(verts)
        for k in range(len(verts)):
            verts[k] = b2Vec2(verts[k]) + pos
        verts = list(map(self.renderer.to_screen, verts))
        #self.renderer.DrawSolidPolygon(verts, col)
        pygame.draw.polygon(surface, col.bytes, verts, 0)

    def clearScreen(self):
        if self.dark:
            self.screen.fill((0, 0, 0))
        else:
            self.screen.fill((255, 255, 255))

    def Step(self, settings):

        rewards = [0.01 for i in self.agents]

        #if self.timer % 10 == 0:
        #    print(self.timer)
        #    print(self.game_over)
        #self.noClear = self.game_ove
        self.pause = 0

        #if self.game_over > 30:
        #    self.pause = 1

        #print("before step")

        Framework.Step(self, settings, pause=self.pause)

        

        if not self.glitch and not self.noClear:
            self.clearScreen()

        #if self.noClear and self.timer % self.noClear == 0:
        #    self.clearScreen()

        

        self.timer += 1

        #if self.viewZoom < 6.95:
        #    self.viewZoom += (7 - self.viewZoom) / 20.
            #self.clearScreen()

        if self.game_over > 70:
            #self.glitch = 0
            self.resetLevel()
            return


        if self.game_over > 0:
            #self.glitch = 1
            self.player = None
            self.game_over += 1

        if self.game_over == 0:
            vel = 2
            if self.pressed[Keys.K_w]:
                self.player.linearVelocity += b2Vec2(0, 1) * vel
            if self.pressed[Keys.K_a]:
                self.player.linearVelocity += b2Vec2(-1, 0) * vel
            if self.pressed[Keys.K_s]:
                self.player.linearVelocity += b2Vec2(0, -1) * vel
            if self.pressed[Keys.K_d]:
                self.player.linearVelocity += b2Vec2(1, 0) * vel

            if self.rMouseDown and self.timer % 7 == 0:
                cut_start, cut_dir = self.getCutLine()
                new_body = laser_cut(self.world, cut_start, cut_dir,
                              laser_half_width=LASER_HALF_WIDTH)


        #if self.game_over == 0:
            #view = self.viewCenter + b2Vec2(7, -12)
            #diff = self.player.position - view
            #if diff.length > 5:
            #    self.viewCenter += diff / 10.0
        #print(view, self.player.position)


        #draw stuff
        


                
                
        """
        tim = self.timer / 15.0
        square = get_poly(4, 3, tim).vertices
        self.drawSolidPolygon(square, b2Color(1, 0, 1), self.exitPos)
        square = get_poly(4, 2, tim * 1.2).vertices
        self.drawSolidPolygon(square, b2Color(0.7, 0, 0.7), self.exitPos)
        square = get_poly(4, 1, tim * 1.44).vertices
        self.drawSolidPolygon(square, b2Color(0.4, 0, 0.4), self.exitPos)
        """

        for idx, ag in enumerate(self.agents):

            #print(idx)
            #print(ag)

            if ag is not None:

                wall_detect = np.zeros(NUM_SIGHT_RAYS)
                en_detect = np.zeros(NUM_SIGHT_RAYS)
                bullet_detect = np.zeros(NUM_SIGHT_RAYS)
                dists = np.zeros(NUM_SIGHT_RAYS)

                for k in range(0, NUM_SIGHT_RAYS):

                    ang = ag.angle - SIGHT_ANGLE / 2. + k * (SIGHT_ANGLE / (NUM_SIGHT_RAYS - 1.))
                    pt, body = get_sight_ray(self.world, ag, ang)

                    if pt is not None:
                        dist = (ag.position - pt).length

                        wall_detect[k] = (body.userData == 'ground')

                        en_detect[k] = isinstance(body.userData, Agent)

                        bullet_detect[k] = (body.userData == DANGER)#isinstance(body.userData, Agent)

                        dists[k] = dist / 30.
                    #print(pt)
                    if pt is not None and self.drawRays:
                        col = b2Color(0, 0, 0)
                        if isinstance(body.userData, Agent):
                            col = b2Color(1, 0, 0)
                        #print(0)
                        #print(body)
                        self.renderer.DrawSegment(self.renderer.to_screen(ag.position), self.renderer.to_screen(pt), col)
                    #else:
                    #    self.DrawSegment(ag.position, pt, b2Color(0, 0, 0))

                x = ag.position.x
                x /= 50.

                y = ag.position.y + 30
                y /= 50.

                ang = ag.angle

                #print(x, y, ang)

                ag.userData.receive_input(np.concatenate((wall_detect, en_detect, bullet_detect, dists)), x, y, ang)


                vel = 2

                x, y, rot_ch, shoot = ag.userData.get_actions()

                #wantAngle = atan2(rot_y, rot_x)

                ag.angle += rot_ch * 0.01


                ag.linearVelocity += b2Vec2(float(x), float(y)) * vel


                curAng = ag.angle

                target = ag.position + b2Vec2(cos(curAng), sin(curAng))

                if shoot > 0. and ag.userData.reload > 1.:

                    print("shooting")

                    self.agentShoot(ag, target)
                    ag.userData.reload = 0.

                if ag.userData.reload <= 1.:
                    ag.userData.reload += 0.01

                """

                targetVec = self.exitPos - ag.position
                l = targetVec.length
                targetVec /= l * l
                targetVec *= 1000000
                if l < 15:
                    ag.ApplyForce(targetVec, ag.position, True)
                    rewards[idx] += 1.
                if self.getDistance(ag.position, self.exitPos) < 5:
                    rewards[idx] += 10.
                    for j in range(len(self.agents)):
                        if j != idx:
                            rewards[j] -= 5.
                    #self.setLevel(self.level + 1)
                """

        m = self.mouseWorld
        plPos = self.player.position

        if m and plPos:
            plToM = m - plPos

            plWantAngle = math.atan2(plToM.y, plToM.x)

            self.player.angle = plWantAngle


        bodies = self.world.bodies

        remove_bodies = []

        body_pairs = [(p['fixtureA'].body, p['fixtureB'].body)
                      for p in self.points]

        explode_bodies = []
        explode_bodies_player = []

        cnt = 0


        for body1, body2 in body_pairs:

            #continue
            
            if not (body1 and body2):
                continue

            if body1.userData == DANGER:
                body1, body2 = body2, body1


            if isinstance(body1.userData, Agent) and body1 != self.player and body2.userData == DANGER:

                #print(body1)
                if body1 not in explode_bodies:
                    explode_bodies.append(body1)
                #body_explode(self.world, body1, get_mass_center(body1), 3, PLAYER_PART)

                body1.userData.hp = 0

                self.agents[body1.userData.id] = None

            if body1.userData == PLAYER_PART:
                body1, body2 = body2, body1

            if body1.userData != DANGER and body2.userData == DANGER:
                if body2 not in remove_bodies:
                    remove_bodies.append(body2)



            cnt += 1

        #print(cnt)



        #print(len(bodies))
        for i in bodies:
            tr = i.transform

            col = b2Color(0, 0, 0)

            if isinstance(i.userData, Agent):
                if i.userData.col is not None:
                    col = i.userData.col

            
            for j in i.fixtures:
                if type(j.shape) != b2PolygonShape and type(j.shape) != b2EdgeShape:
                    continue
                #print(j.shape.vertices)
                verts = list(j.shape.vertices)
                for k in range(len(verts)):
                    verts[k] = b2Mul(tr, verts[k])
                verts = list(map(self.renderer.to_screen, verts))
                #print(verts)
                
                
                self.renderer.DrawSolidPolygon(verts, col)



        #if len(explode_bodies) + len(remove_bodies) > 0:
        #    print(explode_bodies, remove_bodies)

        #print("start exploding")

        for i in explode_bodies:
            if i in remove_bodies:
                remove_bodies.remove(i)
            body_explode(self.world, i, i.position + get_random_vector(1), 3, DANGER)

        for i in explode_bodies_player:
            if i == self.player:
                self.game_over = 1
            body_explode(self.world, i, get_mass_center(i), 3, PLAYER_PART)

        #print("start removing")

        for i in remove_bodies:
            self.world.DestroyBody(i)

        #print("done removing")


        RES = (800, 600)

        #pixels = np.zeros(RES + (3,))

        #print(pixels.shape)

        """

        pixels = np.array(pygame.PixelArray(self.renderer.surface))

        w = pixels.shape[0]
        h = pixels.shape[1]
        w_b = int(w * 0.1)
        h_b = int(h * 0.1)

        #pixels = pixels[w_b:-w_b, h_b:-h_b]

        pixels = resize(pixels, (256, 256))

        if self.pressed[Keys.K_q]:
            print("screenshot")
            plt.imshow(pixels)
            plt.savefig("rlArena/screen.png")
            plt.close()

        #for i in range(RES[0]):
        #    for j in range(RES[1]):
        #        pixels[i][j][:] = np.array(getRGB(int(pixelsInt[i][j])))


        #print(pixels.mean())
        #print(pixels.min())
        #print(pixels.max())
        

        for idx, ag in enumerate(self.agents):
            ag.userData.receive_input(pixels)

        """


        #textsurface = self.myfont.render('Rewards:{}'.format(rewards), False, (0, 0, 0))

        #self.screen.blit(textsurface,(0,0))


        return rewards

        

if __name__ == "__main__":
    main(Arena)
