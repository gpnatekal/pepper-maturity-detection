import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './dashboard.component';
import { authGuard } from '../auth/auth.guard';
import { HomeComponent } from './home/home.component';
import { MaturityComponent } from './maturity/maturity.component';
import { PriceComponent } from './price/price.component';
import { DiseaseComponent } from './disease/disease.component';

const routes: Routes = [
    {
    path: '',
    component: DashboardComponent,
    canActivateChild: [authGuard],
    children: [
      {
        path: '',
        redirectTo: 'home',
        pathMatch: 'full'
      },
      {
        path: 'home',
        component: HomeComponent,
      },
      {
        path: 'maturity',
        component: MaturityComponent
      },
      {
        path: 'price',
        component: PriceComponent
      },
      {
        path: 'disease',
        component: DiseaseComponent
      },
      
    ]
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class DashboardRoutingModule { }
