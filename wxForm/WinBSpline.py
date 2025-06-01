# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 4.2.1-0-g80c4cb6)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

import gettext
_ = gettext.gettext

###########################################################################
## Class MainWindowBase
###########################################################################

class MainWindowBase ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 1000,600 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

        self.SetSizeHints( wx.Size( 1000,600 ), wx.DefaultSize )

        bSizer1 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_panel1 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer2 = wx.BoxSizer( wx.HORIZONTAL )

        self.fig_panel = wx.Panel( self.m_panel1, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        self.fig_sizer = wx.BoxSizer( wx.VERTICAL )


        self.fig_panel.SetSizer( self.fig_sizer )
        self.fig_panel.Layout()
        self.fig_sizer.Fit( self.fig_panel )
        bSizer2.Add( self.fig_panel, 5, wx.EXPAND |wx.ALL, 5 )

        self.m_staticline1 = wx.StaticLine( self.m_panel1, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_VERTICAL )
        bSizer2.Add( self.m_staticline1, 0, wx.EXPAND |wx.ALL, 5 )

        bSizer4 = wx.BoxSizer( wx.VERTICAL )

        self.m_textCtrl1 = wx.TextCtrl( self.m_panel1, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE|wx.TE_READONLY )
        bSizer4.Add( self.m_textCtrl1, 1, wx.ALL|wx.EXPAND, 5 )

        sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( self.m_panel1, wx.ID_ANY, _(u"参数设置") ), wx.VERTICAL )

        bSizer10 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText3 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"控制点数 (n):"), wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText3.Wrap( -1 )

        bSizer10.Add( self.m_staticText3, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

        self.m_textCtrl3 = wx.TextCtrl( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"5"), wx.DefaultPosition, wx.Size( 50,-1 ), wx.TE_CENTER )
        bSizer10.Add( self.m_textCtrl3, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        sbSizer2.Add( bSizer10, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5 )

        bSizer11 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText4 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"端点斜率 (°):"), wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText4.Wrap( -1 )

        bSizer11.Add( self.m_staticText4, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

        self.m_slider2 = wx.Slider( sbSizer2.GetStaticBox(), wx.ID_ANY, 0, -90, 90, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL|wx.SL_LABELS|wx.SL_SELRANGE )
        bSizer11.Add( self.m_slider2, 1, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        sbSizer2.Add( bSizer11, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5 )

        bSizer13 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText5 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"操作模式:"), wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText5.Wrap( -1 )

        bSizer13.Add( self.m_staticText5, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

        bSizer14 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_radioBtn1 = wx.RadioButton( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"新增点"), wx.DefaultPosition, wx.DefaultSize, wx.RB_GROUP )
        bSizer14.Add( self.m_radioBtn1, 0, wx.ALL, 5 )

        self.m_radioBtn2 = wx.RadioButton( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"移动点"), wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_radioBtn2.SetValue( True )
        bSizer14.Add( self.m_radioBtn2, 0, wx.ALL, 5 )


        bSizer13.Add( bSizer14, 1, wx.EXPAND, 5 )


        sbSizer2.Add( bSizer13, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5 )


        bSizer4.Add( sbSizer2, 1, wx.ALL|wx.EXPAND, 5 )

        bSizer12 = wx.BoxSizer( wx.VERTICAL )

        self.m_button1 = wx.Button( self.m_panel1, wx.ID_ANY, _(u"重新绘制"), wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer12.Add( self.m_button1, 0, wx.ALL|wx.ALIGN_RIGHT, 5 )


        bSizer4.Add( bSizer12, 0, wx.EXPAND, 5 )


        bSizer2.Add( bSizer4, 2, wx.ALL|wx.EXPAND, 10 )


        self.m_panel1.SetSizer( bSizer2 )
        self.m_panel1.Layout()
        bSizer2.Fit( self.m_panel1 )
        bSizer1.Add( self.m_panel1, 1, wx.EXPAND, 5 )


        self.SetSizer( bSizer1 )
        self.Layout()
        self.m_statusBar1 = self.CreateStatusBar( 1, wx.STB_SIZEGRIP, wx.ID_ANY )
        self.m_menubar1 = wx.MenuBar( 0 )
        self.SetMenuBar( self.m_menubar1 )


        self.Centre( wx.BOTH )

        # Connect Events
        self.m_slider2.Bind( wx.EVT_SLIDER, self.change_slope )
        self.m_radioBtn1.Bind( wx.EVT_RADIOBUTTON, self.change_mode_add )
        self.m_radioBtn2.Bind( wx.EVT_RADIOBUTTON, self.change_mode_move )
        self.m_button1.Bind( wx.EVT_BUTTON, self.replot )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def change_slope( self, event ):
        event.Skip()

    def change_mode_add( self, event ):
        event.Skip()

    def change_mode_move( self, event ):
        event.Skip()

    def replot( self, event ):
        event.Skip()


